#!/usr/bin/env python3
"""Crisp compile + final run helper.

This script takes a **bufferized** MLIR file (containing memref instead of tensor),
compiles it through the CRISP pipeline (lowering to LLVM dialect, then to
LLVM IR, object file, and finally an executable), and runs the executable.

Supports four configs:
  - crisp:         CRISP optimized ASan instrumentation
  - asan0:         standard ASan instrumentation
  - asan-outline:  ASan with outline instrumentation
  - asan-opt:      ASan with outline instrumentation and opt disabled

python crisp_phase2.py [testcases].mlir --config asan0
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


MLIR_OPT = Path.home() / "torch-mlir/build/bin/mlir-opt"
MLIR_TRANSLATE = Path.home() / "torch-mlir/build/bin/mlir-translate"
LLC = Path.home() / "torch-mlir/build/bin/llc"
BUILD_DIR = MLIR_OPT.parent.parent
ASAN_RT = Path("/usr/lib/llvm-22/lib/clang/22/lib/linux/libclang_rt.asan-x86_64.so")
MLIR_LIBDIR = BUILD_DIR / "lib"
SHARED_LIBS = [
    str(MLIR_LIBDIR / "libmlir_c_runner_utils.so"),
    str(MLIR_LIBDIR / "libmlir_runner_utils.so"),
]
DL = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"


@dataclass
class BenchConfig:
    tag: str
    asan_enabled: bool
    crisp_enabled: bool
    needs_asan_rt: bool


ASAN0_CONFIG = BenchConfig(
    tag="asan0", asan_enabled=True, crisp_enabled=False, needs_asan_rt=True
)

CRISP_CONFIG = BenchConfig(
    tag="crisp", asan_enabled=False, crisp_enabled=True, needs_asan_rt=True
)

ASAN_OUTLINE_CONFIG = BenchConfig(
    tag="asan-outline", asan_enabled=True, crisp_enabled=False, needs_asan_rt=True
)

ASAN_OPT_CONFIG = BenchConfig(
    tag="asan-opt", asan_enabled=True, crisp_enabled=False, needs_asan_rt=True
)

BASE_CONFIG = BenchConfig(
    tag="base", asan_enabled=False, crisp_enabled=False, needs_asan_rt=False
)


def run_cmd(cmd: list, cwd=None, env=None, check=True) -> subprocess.CompletedProcess:
    """Run a shell command and optionally check for errors."""
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"[cmd] {cmd_str}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)
    if result.stdout:
        print(f"[stdout]\n{result.stdout}")
    if result.stderr:
        print(f"[stderr]\n{result.stderr}")
    if check and result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else "(no stderr)"
        raise RuntimeError(f"Subprocess failed (rc={result.returncode}): {stderr[-800:]}")

    return result


def build_phase2_pipeline(asan: bool = False, crisp: bool = False) -> str:
    """Build the MLIR pass pipeline from bufferized MLIR to LLVM dialect."""
    passes = []
    passes.extend(
        [
            f"set-llvm-module-datalayout{{data-layout={DL}}}",
            "func.func(linalg-generalize-named-ops)",
            "func.func(linalg-fuse-elementwise-ops)",
            "convert-shape-to-std",
        ]
    )
    if crisp:
        passes.append("func.func(asan-access-instrument)")
    # passes.extend(
    #     [
    #         "one-shot-bufferize{"
    #         "copy-before-write "
    #         "bufferize-function-boundaries "
    #         "function-boundary-type-conversion=identity-layout-map"
    #         "}",
    #         "func.func(buffer-hoisting)",
    #         "buffer-results-to-out-params",
    #         "drop-equivalent-buffer-results",
    #         "func.func(expand-realloc)",
    #         "buffer-deallocation-pipeline",
    #     ]
    # )
    if crisp:
        passes.append("func.func(asan-lifecycle-instrument)")
    passes.append("func.func(convert-linalg-to-loops)")
    if crisp:
        passes.extend(
            [
                "func.func(asan-static-check, asan-fold-subview)",
                "asan-optimization",
                "canonicalize",
                "cse",
                "asan-check-elimination",
                "canonicalize",
                "cse",
                "asan-hoist-check",
                "func.func(asan-static-check, asan-fold-subview)",
                "canonicalize",
                "cse",
                "asan-check-elimination",
                "canonicalize",
                "cse",
                "func.func(asan-writeback-elimination)",
                "canonicalize",
                "cse",
            ]
        )
    # NOTE: Native ASan is handled by clang -fsanitize=address at the LLVM IR
    # level (functions get sanitize_address attribute + inline shadow checks).
    # We do NOT run MLIR-level asan-* passes here.
    if crisp:
        passes.append("convert-asan-to-llvm")
    passes.extend(
        [
            "func.func(lower-affine)",
            "canonicalize",
            "convert-scf-to-cf",
            "func.func(arith-expand)",
            "func.func(convert-math-to-llvm)",
            "convert-math-to-libm",
            "expand-strided-metadata",
            "finalize-memref-to-llvm",
            "lower-affine",
            "convert-scf-to-cf",
            "convert-bufferization-to-memref",
            "finalize-memref-to-llvm",
            "func.func(convert-arith-to-llvm)",
            "convert-vector-to-llvm",
            "convert-func-to-llvm",
            "convert-cf-to-llvm",
            "convert-complex-to-llvm",
        ]
    )
    passes.extend(
        [
            "reconcile-unrealized-casts",
            "canonicalize",
            "cse",
        ]
    )
    return f"builtin.module({','.join(passes)})"


_HOST_TRIPLE: Optional[str] = None


def _get_host_triple() -> str:
    """Query clang for the host target triple and cache it."""
    global _HOST_TRIPLE
    if _HOST_TRIPLE is None:
        result = subprocess.run(
            ["clang", "-dumpmachine"],
            capture_output=True,
            text=True,
            check=True,
        )
        _HOST_TRIPLE = result.stdout.strip()
    return _HOST_TRIPLE


def _ensure_target_triple(llvm_ir_path: Path):
    """Inject host target triple into .ll if missing so clang doesn't warn."""
    content = llvm_ir_path.read_text()
    if "target triple" in content:
        return
    triple = _get_host_triple()
    triple_line = f'target triple = "{triple}"\n'
    m = re.search(r"^source_filename\s*=.*$", content, re.MULTILINE)
    if m:
        insert_pos = m.end()
        content = content[:insert_pos] + "\n" + triple_line + content[insert_pos:]
    else:
        lines = content.splitlines(keepends=True)
        if lines:
            content = lines[0] + triple_line + "".join(lines[1:])
        else:
            content = triple_line + content
    llvm_ir_path.write_text(content)


def _inject_sanitize_address(llvm_ir_path: Path):
    """Inject sanitize_address attribute into all functions in .ll file."""
    content = llvm_ir_path.read_text()
    if "sanitize_address" in content:
        return

    # 1. 在已有的 attributes #N 中追加 sanitize_address
    attr_pattern = re.compile(r"^attributes\s+#(\d+)\s+=\s+\{([^}]*)\}", re.MULTILINE)
    existing_attrs = list(attr_pattern.finditer(content))
    for m in reversed(existing_attrs):
        attr_body = m.group(2).strip()
        if "sanitize_address" in attr_body:
            continue
        new_body = attr_body + " sanitize_address" if attr_body else "sanitize_address"
        new_str = f"attributes #{m.group(1)} = {{ {new_body} }}"
        content = content[:m.start()] + new_str + content[m.end():]

    # 2. 给没有属性组的 define/declare 函数统一分配新编号
    func_pattern = re.compile(
        r"^(define|declare)\s+[^\n]*?\)\s*(?!\s*#)(?=\s*\{|$)",
        re.MULTILINE,
    )
    max_attr_num = max((int(m.group(1)) for m in existing_attrs), default=-1)
    new_attr_num = max_attr_num + 1
    func_matches = list(func_pattern.finditer(content))
    if func_matches:
        for m in reversed(func_matches):
            matched_text = m.group(0)
            rparen_idx = matched_text.rfind(")")
            absolute_idx = m.start() + rparen_idx + 1
            content = content[:absolute_idx] + f" #{new_attr_num}" + content[absolute_idx:]
        content = content.rstrip("\n") + f"\n\nattributes #{new_attr_num} = {{ sanitize_address }}\n"

    llvm_ir_path.write_text(content)


def _prepare_llvm_ir(llvm_ir_path: Path, add_sanitize: bool = False):
    _ensure_target_triple(llvm_ir_path)
    if add_sanitize:
        _inject_sanitize_address(llvm_ir_path)

def compile_mlir_to_llvm_dialect(
    input_path: Path, output_path: Path, cfg: BenchConfig, debug_ir: bool = False
) -> float:
    """Run mlir-opt to lower bufferized MLIR to LLVM dialect."""
    pipeline = build_phase2_pipeline(
        asan=cfg.asan_enabled,
        crisp=cfg.crisp_enabled,
    )
    cmd = [
        str(MLIR_OPT),
        str(input_path),
        f"--pass-pipeline={pipeline}",
        "-o", str(output_path),
    ]
    if debug_ir:
        cmd.append("--mlir-print-ir-after-all")

    t0 = time.perf_counter()
    run_cmd(cmd)
    elapsed = time.perf_counter() - t0
    return elapsed


def translate_to_llvmir(llvm_dialect_path: Path, llvm_ir_path: Path):
    """Translate LLVM dialect MLIR to LLVM IR (.ll)."""
    run_cmd([
        str(MLIR_TRANSLATE),
        str(llvm_dialect_path),
        "--mlir-to-llvmir",
        "-o", str(llvm_ir_path),
    ])


def compile_llvmir_to_object(
    llvm_ir_path: Path, obj_path: Path, cfg: BenchConfig, opt_level: int = 0
):
    """Compile LLVM IR to object file.

    For native ASan we use clang -fsanitize=address so that clang adds the
    sanitize_address function attribute and runs the LLVM ASan instrumentation
    pass (inline shadow checks by default).
    """
    cmd = [
        "clang",
        f"-O{opt_level}",
    ]
    if cfg.asan_enabled or cfg.tag =="crisp":
        cmd.extend([
            "-fsanitize=address",
            "-fno-omit-frame-pointer",
        ])
        if cfg.tag == "asan-outline":
            cmd.append("-fsanitize-address-outline-instrumentation")
        elif cfg.tag == "asan-opt":
            cmd.extend([
                "-fsanitize-address-outline-instrumentation",
                "-mllvm", "-asan-opt=false",
            ])
    cmd.extend([
        "-c",
        str(llvm_ir_path),
        "-o", str(obj_path),
    ])
    run_cmd(cmd)


def link_executable(obj_path: Path, exe_path: Path, cfg: BenchConfig, opt_level: int = 0):
    """Link object file into an executable with clang."""
    libs = []
    rpaths = []
    for lib in SHARED_LIBS:
        libs.extend(["-l", Path(lib).stem[3:]])  # strip 'lib' prefix
        rpaths.append(f"-Wl,-rpath,{Path(lib).parent}")

    cmd = [
        "clang",
        str(obj_path),
        f"-O{opt_level}",
        "-o", str(exe_path),
        f"-L{MLIR_LIBDIR}",
    ] + libs + rpaths

    if cfg.asan_enabled:
        # Native ASan: clang handles instrumentation, runtime linking,
        # and .init_array registration automatically.
        cmd.append("-fsanitize=address")
    elif cfg.needs_asan_rt:
        # Legacy/CRISP path: manual runtime linking for MLIR-level instrumentation.
        cmd.extend([f"-L{ASAN_RT.parent}", f"-l:{ASAN_RT.name}"])

    run_cmd(cmd)


def run_executable(exe_path: Path, cfg: BenchConfig) -> int:
    """Run the compiled executable and return its exit code."""
    env = os.environ.copy()

    # Set LD_LIBRARY_PATH so shared libs are found at runtime
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    lib_paths = [str(MLIR_LIBDIR)]
    if existing_ld:
        lib_paths.append(existing_ld)
    env["LD_LIBRARY_PATH"] = ":".join(lib_paths)

    if cfg.asan_enabled:
        # Native ASan executable already has the runtime linked/registered;
        # no LD_PRELOAD needed. Just keep leak detection off for consistency.
        env.setdefault("ASAN_OPTIONS", "detect_leaks=0")
    elif cfg.needs_asan_rt:
        # Legacy/CRISP path: preload the shared ASan runtime.
        existing = env.get("LD_PRELOAD", "")
        if "libclang_rt.asan" not in existing:
            env["LD_PRELOAD"] = (str(ASAN_RT) + " " + existing).strip()
        env.setdefault("ASAN_OPTIONS", "detect_leaks=0")

    result = run_cmd([str(exe_path)], env=env, check=False)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Compile bufferized MLIR to executable and run it."
    )
    parser.add_argument("input_mlir", help="Input bufferized MLIR file")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Optional output directory. Default: ./<input_stem>_output",
    )
    parser.add_argument(
        "--config",
        choices=["crisp", "asan0", "asan-outline", "asan-opt", "base"],
        default="crisp",
        help="Select compilation config: 'crisp' for CRISP, 'asan0' for standard ASan, "
             "'asan-outline' for ASan with outline instrumentation, "
             "'asan-opt' for ASan with outline instrumentation and opt disabled, "
             "'base' for plain clang without AddressSanitizer.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep intermediate files (LLVM dialect, LLVM IR, object file)",
    )
    parser.add_argument("--debug-ir", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--opt",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Optimization level for llc when compiling LLVM IR to object (default: 0).",
    )

    args = parser.parse_args()

    input_path = Path(args.input_mlir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.keep:
        if args.output_dir is None:
            output_dir = Path.cwd() / f"{input_path.stem}_output"
        else:
            output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        # 不保留任何中间文件/目录，全部在临时目录中操作
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(temp_dir.name)

    config_map = {
        "crisp": CRISP_CONFIG,
        "asan0": ASAN0_CONFIG,
        "asan-outline": ASAN_OUTLINE_CONFIG,
        "asan-opt": ASAN_OPT_CONFIG,
        "base": BASE_CONFIG,
    }
    cfg = config_map[args.config]

    # Intermediate files
    llvm_dialect_path = output_dir / f"{input_path.stem}_llvm.mlir"
    llvm_ir_path = output_dir / f"{input_path.stem}.ll"
    obj_path = output_dir / f"{input_path.stem}.o"
    exe_path = output_dir / input_path.stem

    try:
        print(f"[{cfg.tag}] Compiling bufferized MLIR -> LLVM dialect ...")
        elapsed = compile_mlir_to_llvm_dialect(
            input_path, llvm_dialect_path, cfg, debug_ir=args.debug_ir
        )
        print(f"[{cfg.tag}] MLIR lowering done in {elapsed:.4f} s")

        print(f"[{cfg.tag}] Translating LLVM dialect -> LLVM IR ...")
        translate_to_llvmir(llvm_dialect_path, llvm_ir_path)
        _prepare_llvm_ir(llvm_ir_path, add_sanitize=cfg.asan_enabled)

        print(f"[{cfg.tag}] Compiling LLVM IR -> object file ...")
        compile_llvmir_to_object(llvm_ir_path, obj_path, cfg, opt_level=args.opt)

        print(f"[{cfg.tag}] Linking executable: {exe_path}")
        link_executable(obj_path, exe_path, cfg, opt_level=args.opt)

        print(f"[{cfg.tag}] Executing {exe_path} ...")
        exit_code = run_executable(exe_path, cfg)
        print(f"[{cfg.tag}] Exit code: {exit_code}")

        return exit_code
    finally:
        if not args.keep:
            print(f"[{cfg.tag}] Cleaning up all artifacts ...")
            temp_dir.cleanup()


if __name__ == "__main__":
    sys.exit(main())

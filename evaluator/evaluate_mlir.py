#!/usr/bin/env python3
"""
evaluate_mlir.py - MSET-compatible evaluation for MLIR pass correctness.

Evaluates MLIR test cases using crisp_phase2.py toolchain and outputs
results in the same format as the original MSET C++ evaluator.

Supports parallel execution.

python evaluate_mlir.py --table
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants matching test_case_information.h
# ---------------------------------------------------------------------------

REGIONS = ["heap", "stack", "global"]
REGIONS_INFO = ["Heap", "Stack", "Global"]

TEMPORAL_BUG_TYPES = ["misuse_of_free", "double_free", "use_after_star"]
TEMPORAL_BUGS_INFO = ["Misuse-of-free", "Double-free", "Use-after-*"]

TEMPORAL_MEMORY_STATES = ["used_memory", "freed_memory"]
TEMPORAL_MEMORY_STATES_INFO = ["(Re)used Memory", "Freed Memory"]

SPATIAL_BUG_TYPES = ["linear_ooba", "non_linear_ooba", "type_confusion_ooba"]
SPATIAL_BUGS_INFO = ["Linear OOBA", "Non-Linear OOBA", "Type Confusion OOBA"]

ORIGIN_TARGET_RELATIONS = ["non_object", "intra_object", "inter_object"]
ORIGIN_TARGET_RELATIONS_INFO = ["Non-Object", "Intra-Object", "Inter-Object"]

FLOWS = ["overflow", "underflow"]
FLOWS_INFO = ["Overflow", "Underflow"]

ACCESS_LOCATIONS = ["stdlib", "direct"]
ACCESS_LOCATIONS_INFO = ["Stdlib", "Direct"]

ACCESS_ACTIONS = ["read", "write"]
ACCESS_ACTIONS_INFO = ["Read", "Write"]

# Bug type aliases for user-friendly --bug-type filtering.
# Maps common lowercase/no-underscore variants to the canonical display name.
BUG_TYPE_ALIASES: Dict[str, str] = {
    # Spatial
    "linear_ooba": "Linear OOBA",
    "linearooba": "Linear OOBA",
    "linear": "Linear OOBA",
    "non_linear_ooba": "Non-Linear OOBA",
    "nonlinearooba": "Non-Linear OOBA",
    "nonlinear": "Non-Linear OOBA",
    "type_confusion_ooba": "Type Confusion OOBA",
    "typeconfusion": "Type Confusion OOBA",
    "typeconfusionooba": "Type Confusion OOBA",
    # Temporal
    "misuse_of_free": "Misuse-of-free",
    "misuseoffree": "Misuse-of-free",
    "double_free": "Double-free",
    "doublefree": "Double-free",
    "use_after_star": "Use-after-*",
    "useafterstar": "Use-after-*",
}

_ALL_BUG_TYPE_NAMES = set(BUG_TYPE_ALIASES.values())


def normalize_bug_type(raw: str) -> Optional[str]:
    """Convert user input to canonical bug type name, or None if invalid."""
    key = raw.strip().lower()
    if key in BUG_TYPE_ALIASES:
        return BUG_TYPE_ALIASES[key]
    # Also accept exact display names (case-insensitive)
    for canonical in _ALL_BUG_TYPE_NAMES:
        if canonical.lower() == key:
            return canonical
    return None


# ---------------------------------------------------------------------------
# Result enum matching MSET exec_result_t
# ---------------------------------------------------------------------------

class ExecResult(Enum):
    PRECONDITIONS_FAILED = auto()
    DETECTED = auto()
    DETECTED_SIGSEGV = auto()
    UNDETECTED_TIMEOUT = auto()
    INVALID = auto()
    UNDETECTED = auto()

    def __str__(self):
        mapping = {
            ExecResult.PRECONDITIONS_FAILED: "PRECONDITIONS FAILED",
            ExecResult.DETECTED: "DETECTED",
            ExecResult.DETECTED_SIGSEGV: "DETECTED_SIGSEGV",
            ExecResult.UNDETECTED_TIMEOUT: "UNDETECTED_TIMEOUT",
            ExecResult.INVALID: "INVALID",
            ExecResult.UNDETECTED: "UNDETECTED",
        }
        return mapping[self]


def overall_result_to_string(result: ExecResult) -> str:
    if result in (ExecResult.DETECTED_SIGSEGV, ExecResult.UNDETECTED_TIMEOUT):
        # MSET C++ evaluator treats these as "ERROR: UNKNOWN" in overall_result_to_string,
        # but they are valid internal states. For output we map them meaningfully.
        if result == ExecResult.DETECTED_SIGSEGV:
            return "DETECTED_SIGSEGV"
        return "UNDETECTED_TIMEOUT"
    return str(result)


# ---------------------------------------------------------------------------
# Test case information classes
# ---------------------------------------------------------------------------

@dataclass
class TestCaseInformation:
    file_name: str
    file_name_without_suffix: str
    file_path: str
    is_validation: bool
    variant_number: int
    as_string: str = ""
    key: str = ""

    def get_test_case_key(self) -> str:
        return self.key

    def get_file_name(self) -> str:
        return self.file_name

    def get_file_name_without_suffix(self) -> str:
        return self.file_name_without_suffix

    def get_file_path(self) -> str:
        return self.file_path

    def get_is_validation(self) -> bool:
        return self.is_validation

    def get_variant_number(self) -> int:
        return self.variant_number

    def __str__(self):
        return self.as_string


@dataclass
class TemporalTestCaseInformation(TestCaseInformation):
    region_name: str = ""
    temporal_bug_name: str = ""
    temporal_memory_state_name: str = ""
    access_location_name: str = ""
    access_action_name: str = ""

    def __post_init__(self):
        self.as_string = (
            f"{self.temporal_bug_name} {self.access_location_name} {self.access_action_name} on "
            f"{self.region_name} {self.temporal_memory_state_name}, variant {self.variant_number}"
        )
        self.key = (
            f"{self.temporal_bug_name} {self.access_location_name} {self.access_action_name} "
            f"{self.region_name} {self.temporal_memory_state_name}"
        )


@dataclass
class SpatialTestCaseInformation(TestCaseInformation):
    origin_name: str = ""
    target_name: str = ""
    origin_target_relation_name: str = ""
    spatial_bug_name: str = ""
    flow_name: str = ""
    access_location_name: str = ""
    access_action_name: str = ""

    def __post_init__(self):
        self.as_string = (
            f"{self.origin_target_relation_name} {self.spatial_bug_name} {self.flow_name} "
            f"{self.access_location_name} {self.access_action_name} on ("
            f"{self.origin_name}, {self.target_name}), variant {self.variant_number}"
        )
        self.key = (
            f"{self.origin_target_relation_name} {self.spatial_bug_name} {self.flow_name} "
            f"{self.access_location_name} {self.access_action_name} "
            f"{self.origin_name} {self.target_name}"
        )


# ---------------------------------------------------------------------------
# File name parsing (translated from test_case_information.cpp)
# ---------------------------------------------------------------------------

def find_prefix(s: str, prefixes: List[str]) -> Tuple[int, str]:
    for i, prefix in enumerate(prefixes):
        if s.startswith(prefix):
            return i, s[len(prefix):]
    return -1, s


def construct_from_file_name(file_name: str, file_path: str) -> Optional[TestCaseInformation]:
    remaining = file_name

    # Try temporal bug types
    found_at, remaining = find_prefix(remaining, TEMPORAL_BUG_TYPES)
    if found_at != -1:
        temporal_bug = TEMPORAL_BUGS_INFO[found_at]

        if not remaining or remaining[0] != '_':
            print(f"Unsupported file name '{file_name}'. Expected '_' before {remaining}")
            return None
        remaining = remaining[1:]

        found_at, remaining = find_prefix(remaining, TEMPORAL_MEMORY_STATES)
        if found_at == -1:
            print(f"Unsupported file name '{file_name}'. Expected temporal memory state before {remaining}")
            return None
        temporal_memory_state = TEMPORAL_MEMORY_STATES_INFO[found_at]

        if not remaining or remaining[0] != '_':
            print(f"Unsupported file name '{file_name}'. Expected '_' before {remaining}")
            return None
        remaining = remaining[1:]

        found_at, remaining = find_prefix(remaining, REGIONS)
        if found_at == -1:
            print(f"Unsupported file name '{file_name}'. Expected region before {remaining}")
            return None
        region = REGIONS_INFO[found_at]

        if not remaining or remaining[0] != '_':
            print(f"Unsupported file name '{file_name}'. Expected '_' before {remaining}")
            return None
        remaining = remaining[1:]

        found_at, remaining = find_prefix(remaining, ACCESS_LOCATIONS)
        if found_at == -1:
            print(f"Unsupported file name '{file_name}'. Expected access location before {remaining}")
            return None
        access_location = ACCESS_LOCATIONS_INFO[found_at]

        if not remaining or remaining[0] != '_':
            print(f"Unsupported file name '{file_name}'. Expected '_' before {remaining}")
            return None
        remaining = remaining[1:]

        found_at, remaining = find_prefix(remaining, ACCESS_ACTIONS)
        if found_at == -1:
            print(f"Unsupported file name '{file_name}'. Expected access action before {remaining}")
            return None
        access_action = ACCESS_ACTIONS_INFO[found_at]

        if not remaining or remaining[0] != '_':
            print(f"Unsupported file name '{file_name}'. Expected '_' before {remaining}")
            return None
        remaining = remaining[1:]

        is_validation = False
        if remaining and not remaining[0].isdigit():
            if remaining.startswith("validation"):
                is_validation = True
                remaining = remaining[len("validation"):]
                if not remaining or remaining[0] != '_':
                    print(f"Unsupported file name '{file_name}'. Expected '_' before {remaining}")
                    return None
                remaining = remaining[1:]
            else:
                print(f"Unsupported file name '{file_name}'. Expected variant number or 'validation' before {remaining}")
                return None

        if not remaining or not remaining[0].isdigit():
            print(f"Unsupported file name '{file_name}'. Expected variant number before {remaining}")
            return None

        variant_number = 0
        idx = 0
        while idx < len(remaining) and remaining[idx].isdigit():
            variant_number = variant_number * 10 + int(remaining[idx])
            idx += 1
        remaining = remaining[idx:]

        if remaining and remaining != ".mlir":
            print(f"Unsupported file name '{file_name}'. Must end with '.mlir'. Got {remaining}")
            return None

        file_name_without_suffix = file_name[:-len(".mlir")] if remaining == ".mlir" else file_name

        return TemporalTestCaseInformation(
            file_name=file_name,
            file_name_without_suffix=file_name_without_suffix,
            file_path=file_path,
            is_validation=is_validation,
            variant_number=variant_number,
            region_name=region,
            temporal_bug_name=temporal_bug,
            temporal_memory_state_name=temporal_memory_state,
            access_location_name=access_location,
            access_action_name=access_action,
        )

    # Try spatial bug types
    found_at, remaining = find_prefix(remaining, SPATIAL_BUG_TYPES)
    if found_at == -1:
        return None

    spatial_bug = SPATIAL_BUGS_INFO[found_at]

    if not remaining or remaining[0] != '_':
        print(f"Unsupported file name '{file_name}'. Expected '_' before {remaining}")
        return None
    remaining = remaining[1:]

    found_at, remaining = find_prefix(remaining, REGIONS)
    if found_at == -1:
        print(f"Unsupported file name '{file_name}'. Expected spatial origin before {remaining}")
        return None
    origin = REGIONS_INFO[found_at]

    if not remaining or remaining[0] != '_':
        print(f"Unsupported file name '{file_name}'. Expected '_' before {remaining}")
        return None
    remaining = remaining[1:]

    found_at, remaining = find_prefix(remaining, REGIONS)
    if found_at == -1:
        print(f"Unsupported file name '{file_name}'. Expected spatial target before {remaining}")
        return None
    target = REGIONS_INFO[found_at]

    if not remaining or remaining[0] != '_':
        print(f"Unsupported file name '{file_name}'. Expected '_' before {remaining}")
        return None
    remaining = remaining[1:]

    found_at, remaining = find_prefix(remaining, ORIGIN_TARGET_RELATIONS)
    if found_at == -1:
        print(f"Unsupported file name '{file_name}'. Expected origin-target relation before {remaining}")
        return None
    origin_target_relation = ORIGIN_TARGET_RELATIONS_INFO[found_at]

    if not remaining or remaining[0] != '_':
        print(f"Unsupported file name '{file_name}'. Expected '_' before {remaining}")
        return None
    remaining = remaining[1:]

    found_at, remaining = find_prefix(remaining, FLOWS)
    if found_at == -1:
        print(f"Unsupported file name '{file_name}'. Expected flow name before {remaining}")
        return None
    flow = FLOWS_INFO[found_at]

    if not remaining or remaining[0] != '_':
        print(f"Unsupported file name '{file_name}'. Expected '_' before {remaining}")
        return None
    remaining = remaining[1:]

    found_at, remaining = find_prefix(remaining, ACCESS_LOCATIONS)
    if found_at == -1:
        print(f"Unsupported file name '{file_name}'. Expected access location before {remaining}")
        return None
    access_location = ACCESS_LOCATIONS_INFO[found_at]

    if not remaining or remaining[0] != '_':
        print(f"Unsupported file name '{file_name}'. Expected '_' before {remaining}")
        return None
    remaining = remaining[1:]

    found_at, remaining = find_prefix(remaining, ACCESS_ACTIONS)
    if found_at == -1:
        print(f"Unsupported file name '{file_name}'. Expected access action before {remaining}")
        return None
    access_action = ACCESS_ACTIONS_INFO[found_at]

    if not remaining or remaining[0] != '_':
        print(f"Unsupported file name '{file_name}'. Expected '_' before {remaining}")
        return None
    remaining = remaining[1:]

    is_validation = False
    if remaining and not remaining[0].isdigit():
        if remaining.startswith("validation"):
            is_validation = True
            remaining = remaining[len("validation"):]
            if not remaining or remaining[0] != '_':
                print(f"Unsupported file name '{file_name}'. Expected '_' before {remaining}")
                return None
            remaining = remaining[1:]
        else:
            print(f"Unsupported file name '{file_name}'. Expected variant number or 'validation' before {remaining}")
            return None

    if not remaining or not remaining[0].isdigit():
        print(f"Unsupported file name '{file_name}'. Expected variant number before {remaining}")
        return None

    variant_number = 0
    idx = 0
    while idx < len(remaining) and remaining[idx].isdigit():
        variant_number = variant_number * 10 + int(remaining[idx])
        idx += 1
    remaining = remaining[idx:]

    if remaining and remaining != ".mlir":
        print(f"Unsupported file name '{file_name}'. Must end with '.mlir'. Got {remaining}")
        return None

    file_name_without_suffix = file_name[:-len(".mlir")] if remaining == ".mlir" else file_name

    return SpatialTestCaseInformation(
        file_name=file_name,
        file_name_without_suffix=file_name_without_suffix,
        file_path=file_path,
        is_validation=is_validation,
        variant_number=variant_number,
        origin_name=origin,
        target_name=target,
        origin_target_relation_name=origin_target_relation,
        spatial_bug_name=spatial_bug,
        flow_name=flow,
        access_location_name=access_location,
        access_action_name=access_action,
    )


# ---------------------------------------------------------------------------
# Execution layer
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
CRISP_PHASE2 = SCRIPT_DIR / "crisp_phase2.py"


def is_script_error(stderr: str) -> bool:
    """Heuristic: detect if crisp_phase2.py itself failed (compile error etc.)."""
    indicators = [
        "Traceback (most recent call last):",
        "RuntimeError:",
        "Command failed",
    ]
    return any(ind in stderr for ind in indicators)


def is_killed_by_signal(returncode: int) -> bool:
    return returncode < 0


def signal_name(returncode: int) -> str:
    import signal
    sig_num = -returncode
    return signal.Signals(sig_num).name if hasattr(signal, "Signals") else f"SIG{sig_num}"


def run_single_test(mlir_path: str, config: str, output_dir: str, timeout: int, opt_level: int = 0) -> Tuple[int, bool, str, str]:
    """
    Run crisp_phase2.py on a single MLIR file.
    Returns (returncode, timed_out, stdout_output, stderr_output).
    """
    cmd = [
        sys.executable,
        str(CRISP_PHASE2),
        mlir_path,
        output_dir,
        "--config",
        config,
    ]
    if opt_level != 0:
        cmd.extend(["--opt", str(opt_level)])
    env = os.environ.copy()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return result.returncode, False, result.stdout, result.stderr
    except subprocess.TimeoutExpired as e:
        stdout_out = e.stdout.decode() if e.stdout else ""
        stderr_out = e.stderr.decode() if e.stderr else ""
        return -1, True, stdout_out, stderr_out



# Detection keywords that indicate a bug was caught (in stdout or stderr).
_DETECTION_KEYWORDS = {
    "Byte underflow:",
    "Byte overflow:",
    "Range byte underflow:",
    "Range byte overflow:",
    "is out of logical bounds",
    "is out of bounds",
    "Element type mismatch",
    "double free:",
    "use-after-free:",
    "CHECK failed:"
}


def _contains_detection_keyword(text: str) -> bool:
    return any(kw in text for kw in _DETECTION_KEYWORDS)


def classify_result(returncode: int, timed_out: bool, stdout: str, stderr: str) -> ExecResult:
    """Map crisp_phase2.py return code to ExecResult."""
    # Check both stdout and stderr because crisp_phase2.py may re-print child stderr into its stdout
    if _contains_detection_keyword(stdout) or _contains_detection_keyword(stderr):
        # print("[√] MLIR pass detected out-of-bounds access at compile time.")
        return ExecResult.DETECTED

    if timed_out:
        return ExecResult.UNDETECTED_TIMEOUT
    # Process was killed by a signal (e.g., OOM killer -> SIGKILL)
    if is_killed_by_signal(returncode):
        sig = signal_name(returncode)
        if sig == "SIGSEGV":
            return ExecResult.DETECTED_SIGSEGV
        # SIGKILL (OOM), SIGTERM, etc. -> environment issue, not a detection
        return ExecResult.PRECONDITIONS_FAILED

    # Script-level failure (compile error, missing tool, etc.)
    if is_script_error(stderr):
        return ExecResult.PRECONDITIONS_FAILED

    if returncode == 43:
        return ExecResult.PRECONDITIONS_FAILED

    if returncode == 42 or returncode == 0:
        return ExecResult.UNDETECTED

    # Any other non-zero exit code -> ASan / sanitizer reported an error
    return ExecResult.DETECTED


# ---------------------------------------------------------------------------
# Result collection helpers
# ---------------------------------------------------------------------------

def compute_overall_result(results: List[ExecResult]) -> ExecResult:
    """Compute overall result from a list of variant results (MSET logic)."""
    has_invalid = any(r == ExecResult.INVALID for r in results)
    has_undetected = any(r in (ExecResult.UNDETECTED, ExecResult.UNDETECTED_TIMEOUT) for r in results)
    has_detection = any(r in (ExecResult.DETECTED, ExecResult.DETECTED_SIGSEGV) for r in results)

    if has_invalid:
        return ExecResult.INVALID
    if has_undetected:
        return ExecResult.UNDETECTED
    if has_detection:
        return ExecResult.DETECTED
    return ExecResult.PRECONDITIONS_FAILED


@dataclass
class Counters:
    precond_failed: int = 0
    detections: int = 0
    undetected: int = 0
    invalids: int = 0

    @property
    def total(self) -> int:
        return self.precond_failed + self.detections + self.undetected + self.invalids


def compute_counters(results: List[ExecResult]) -> Counters:
    c = Counters()
    for r in results:
        if r == ExecResult.INVALID:
            c.invalids += 1
        elif r == ExecResult.PRECONDITIONS_FAILED:
            c.precond_failed += 1
        elif r in (ExecResult.UNDETECTED, ExecResult.UNDETECTED_TIMEOUT):
            c.undetected += 1
        else:
            c.detections += 1
    return c


def score_to_str(value: float) -> str:
    return f"{value:.2f}%"


# ---------------------------------------------------------------------------
# Output helpers (matching MSET format)
# ---------------------------------------------------------------------------

def _format_metric(counters: Counters, getter) -> str:
    if counters.total == 0:
        return "N/A"
    val = getter(counters)
    pct = score_to_str(val * 100.0 / counters.total)
    return f"{pct} ({int(val)})"


def _print_category_table(title: str, categories: List[Tuple[str, List[ExecResult]]]):
    """Print a formatted table of detection stats per category."""
    print(f"\n{title}")
    header = f"{'Category':<24} | {'Detection Rate':>14} | {'Precond Failed':>14} | {'Detected':>10} | {'Undetected':>10}"
    print(header)
    print("-" * len(header))
    for name, results in categories:
        c = compute_counters(results)
        det_rate = _format_metric(c, lambda x: x.precond_failed + x.detections)
        precond = _format_metric(c, lambda x: x.precond_failed)
        detected = _format_metric(c, lambda x: x.detections)
        undetected = _format_metric(c, lambda x: x.undetected + x.invalids)
        print(f"{name:<24} | {det_rate:>14} | {precond:>14} | {detected:>10} | {undetected:>10}")


def print_results(results: List[ExecResult]):
    counters = compute_counters(results)
    if not results:
        overall = precond = detected = undetected = "N/A"
    else:
        overall = score_to_str(
            (counters.precond_failed + counters.detections) * 100.0 / counters.total
        )
        precond = score_to_str(counters.precond_failed * 100.0 / counters.total)
        detected = score_to_str(counters.detections * 100.0 / counters.total)
        undetected = score_to_str(
            (counters.undetected + counters.invalids) * 100.0 / counters.total
        )

    print(
        f"Detection rate: {overall} ({counters.precond_failed + counters.detections} out of {counters.total} test cases)"
    )

    print("Results for test cases:")
    print(f"- Preconditions failed: {precond} ({counters.precond_failed})")
    print(f"- Detected: {detected} ({counters.detections})")

    if counters.invalids == 0:
        print(f"- Undetected: {undetected} ({counters.undetected})")
    else:
        print(
            f"- Undetected: {undetected} ({counters.undetected} undetected attempts, {counters.invalids} didn't pass validation)"
        )



def process_results(
    raw_temporal: Dict[str, List[ExecResult]],
    raw_spatial: Dict[str, List[ExecResult]],
    raw_overall: List[ExecResult],
    verbose: bool,
    skipped: int = 0,
):
    if verbose:
        temporal_categories = [
            ("Overall", raw_temporal.get("all", [])),
        ] + [
            (info, raw_temporal.get(info, [])) for info in TEMPORAL_BUGS_INFO
        ] + [
            (info, raw_temporal.get(info, [])) for info in REGIONS_INFO
        ] + [
            (info, raw_temporal.get(info, [])) for info in TEMPORAL_MEMORY_STATES_INFO
        ] + [
            (info, raw_temporal.get(info, [])) for info in ACCESS_LOCATIONS_INFO
        ] + [
            (info, raw_temporal.get(info, [])) for info in ACCESS_ACTIONS_INFO
        ]
        _print_category_table("Temporal Bugs Distribution", temporal_categories)

        spatial_categories = [
            ("Overall", raw_spatial.get("all", [])),
        ] + [
            (info, raw_spatial.get(info, [])) for info in SPATIAL_BUGS_INFO
        ] + [
            (f"Origin {info}", raw_spatial.get("origin " + info, [])) for info in REGIONS_INFO
        ] + [
            (f"Target {info}", raw_spatial.get("target " + info, [])) for info in REGIONS_INFO
        ] + [
            (info, raw_spatial.get(info, [])) for info in ORIGIN_TARGET_RELATIONS_INFO
        ] + [
            (info, raw_spatial.get(info, [])) for info in FLOWS_INFO
        ] + [
            (info, raw_spatial.get(info, [])) for info in ACCESS_LOCATIONS_INFO
        ] + [
            (info, raw_spatial.get(info, [])) for info in ACCESS_ACTIONS_INFO
        ]
        _print_category_table("Spatial Bugs Distribution", spatial_categories)

    print("Overall results:")
    print_results(raw_overall)
    if skipped > 0:
        print(
        f"Skipped {skipped} misuse-of-free variants: deallocating a view is illegal in MLIR.")




# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def evaluate_all(
    test_cases_dir: str,
    config: str,
    timeout: int,
    jobs: int,
    verbose: bool,
    keep: bool,
    opt_level: int = 0,
    bug_types: Optional[List[str]] = None,
) -> Tuple[
    Dict[str, List[ExecResult]],
    Dict[str, List[ExecResult]],
    List[ExecResult],
    int,
]:
    """
    Returns:
      (raw_temporal, raw_spatial, raw_overall)
    """
    test_dir = Path(test_cases_dir)
    mlir_files = sorted(test_dir.glob("*.mlir"))
    if not mlir_files:
        print(f"ERROR: No .mlir test files found in {test_cases_dir}. Aborting.")
        sys.exit(1)

    # Validate and canonicalize bug_types early
    canonical_bug_types: Optional[List[str]] = None
    if bug_types:
        canonical_bug_types = []
        for bt in bug_types:
            normalized = normalize_bug_type(bt)
            if normalized is None:
                print(f"ERROR: Unknown bug type '{bt}'. Supported bug types:")
                for name in sorted(_ALL_BUG_TYPE_NAMES):
                    print(f"  - {name}")
                sys.exit(1)
            canonical_bug_types.append(normalized)
        canonical_bug_types = list(dict.fromkeys(canonical_bug_types))  # dedup preserve order
        print(f"Filtering by bug types: {', '.join(canonical_bug_types)}")

    # Parse and group by test case key
    grouped: Dict[str, List[TestCaseInformation]] = defaultdict(list)
    skipped_variants = 0
    filtered_out = 0
    for f in mlir_files:
        tc = construct_from_file_name(f.name, str(f))
        if tc is None:
            continue
        if isinstance(tc, TemporalTestCaseInformation) and tc.temporal_bug_name == "Misuse-of-free":
            skipped_variants += 1
            continue
        # Apply bug-type filter
        if canonical_bug_types is not None:
            if isinstance(tc, TemporalTestCaseInformation):
                if tc.temporal_bug_name not in canonical_bug_types:
                    filtered_out += 1
                    continue
            elif isinstance(tc, SpatialTestCaseInformation):
                if tc.spatial_bug_name not in canonical_bug_types:
                    filtered_out += 1
                    continue
        grouped[tc.get_test_case_key()].append(tc)

    # Sort variants: validation first, then by variant number
    for key in grouped:
        grouped[key].sort(key=lambda x: (not x.is_validation, x.variant_number, x.key))

    total_groups = len(grouped)
    total_variants = sum(len(v) for v in grouped.values())
    print(f"Found {total_groups} test cases, {total_variants} variants.")
    if filtered_out > 0:
        print(f"Filtered out {filtered_out} variants by bug type.")
    if skipped_variants > 0:
        print(
        f"{skipped_variants} misuse‑of‑free variants skipped")

    # Build work items
    tmp_base = Path(tempfile.gettempdir()) / "mset_mlir_eval"
    tmp_base.mkdir(parents=True, exist_ok=True)

    work_items = []  # (config, tc, out_dir)
    for key, variants in grouped.items():
        for tc in variants:
            out_dir = str(tmp_base / f"{config}_{tc.file_name_without_suffix}")
            work_items.append((config, tc, out_dir))

    print(f"Total work items: {len(work_items)}")

    # Execute in parallel
    results_map: Dict[Tuple[str, str], Tuple[int, bool, str, str]] = {}

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        future_to_work = {}
        for config, tc, out_dir in work_items:
            future = executor.submit(run_single_test, tc.file_path, config, out_dir, timeout, opt_level)
            future_to_work[future] = (config, tc)

        completed = 0
        for future in as_completed(future_to_work):
            config, tc = future_to_work[future]
            try:
                returncode, timed_out, stdout, stderr = future.result()
            except Exception as e:
                print(f"ERROR executing {tc.file_path} with {config}: {e}")
                returncode, timed_out, stdout, stderr = -1, False, str(e), ""

            results_map[(config, tc.file_path)] = (returncode, timed_out, stdout, stderr)
            completed += 1

    # Collect results
    raw_temporal_results: Dict[str, List[ExecResult]] = defaultdict(list)
    raw_spatial_results: Dict[str, List[ExecResult]] = defaultdict(list)
    raw_overall_results: List[ExecResult] = []

    evaluated_variants = 0
    for key, variants in grouped.items():
        group_results: List[ExecResult] = []

        for tc in variants:
            if tc.is_validation:
                rc, to, stdout, stderr = results_map.get((config, tc.file_path), (-1, False, "", ""))
                res = classify_result(rc, to, stdout, stderr)
                # Validation must be UNDETECTED; anything else invalidates the test case
                if res != ExecResult.UNDETECTED:
                    group_results.append(ExecResult.INVALID)
            else:
                rc_test, to_test, stdout_test, stderr_test = results_map.get(
                    (config, tc.file_path), (-1, False, "", "")
                )
                res_test = classify_result(rc_test, to_test, stdout_test, stderr_test)

                group_results.append(res_test)
                evaluated_variants += 1

        overall = compute_overall_result(group_results)

        tc0 = variants[0]
        if isinstance(tc0, TemporalTestCaseInformation):
            t = tc0
            raw_temporal_results[t.temporal_bug_name].append(overall)
            raw_temporal_results[t.temporal_memory_state_name].append(overall)
            raw_temporal_results[t.region_name].append(overall)
            raw_temporal_results[t.access_location_name].append(overall)
            raw_temporal_results[t.access_action_name].append(overall)
            raw_temporal_results["all"].append(overall)
            raw_overall_results.append(overall)

            if overall != ExecResult.DETECTED:
                print(
                    f"For {t.get_test_case_key()}, the overall result is {overall_result_to_string(overall)}"
                )
        else:
            s = tc0
            raw_spatial_results["origin " + s.origin_name].append(overall)
            raw_spatial_results["target " + s.target_name].append(overall)
            raw_spatial_results[s.origin_target_relation_name].append(overall)
            raw_spatial_results[s.flow_name].append(overall)
            raw_spatial_results[s.spatial_bug_name].append(overall)
            raw_spatial_results[s.access_location_name].append(overall)
            raw_spatial_results[s.access_action_name].append(overall)
            raw_spatial_results["all"].append(overall)
            raw_overall_results.append(overall)

            if overall != ExecResult.DETECTED:
                print(
                    f"For {s.get_test_case_key()}, the overall result is {overall_result_to_string(overall)}"
                )

    print(f"Evaluated {total_groups} test cases, {evaluated_variants} variants.")

    if not keep:
        shutil.rmtree(tmp_base, ignore_errors=True)

    return (
        raw_temporal_results,
        raw_spatial_results,
        raw_overall_results,
        skipped_variants,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MLIR pass correctness in MSET-compatible format."
    )
    parser.add_argument(
        "--test-cases-dir",
        default="test_cases_mlir",
        help="Directory containing .mlir test cases (default: test_cases_mlir)",
    )
    parser.add_argument(
        "--config",
        choices=["crisp", "asan0", "asan-outline", "asan-opt"],
        default="crisp",
        help="Compilation config passed to crisp_phase2.py (default: crisp)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Parallel jobs (default: CPU count)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=500,
        help="Timeout per test in seconds (default: 500)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output with per-category breakdown",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep generated binaries in temp directory",
    )
    parser.add_argument(
        "--opt",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Optimization level passed to crisp_phase2.py (default: 0)",
    )
    parser.add_argument(
        "--bug",
        nargs="+",
        default=None,
        help=(
            "Only run test cases matching the given bug type(s). "
            "Examples: --bug linear_ooba type_confusion_ooba, "
            "--bug linearooba, --bug Double-free"
        ),
    )
    args = parser.parse_args()

    (
        raw_temporal,
        raw_spatial,
        raw_overall,
        skipped,
    ) = evaluate_all(
        args.test_cases_dir,
        args.config,
        args.timeout,
        args.jobs,
        args.verbose,
        args.keep,
        args.opt,
        args.bug,
    )

    process_results(
        raw_temporal,
        raw_spatial,
        raw_overall,
        args.verbose,
        skipped,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
PNG to PDF Converter

Usage:
    python png_to_pdf.py <input> [output] [options]

Arguments:
    input       Input PNG file, directory, or wildcard pattern
    output      Output PDF file (optional, auto-generated if omitted)

Options:
    --merge     Merge multiple PNGs into a single PDF (default: one PDF per PNG)
    --quality   JPEG quality for non-RGB images (1-95, default: 95)
    --dpi       DPI for output PDF (default: 72)

Examples:
    # Convert single PNG
    python png_to_pdf.py image.png

    # Convert and specify output name
    python png_to_pdf.py image.png output.pdf

    # Convert all PNGs in a directory (separate PDFs)
    python png_to_pdf.py ./photos/

    # Merge all PNGs into one PDF
    python png_to_pdf.py ./photos/ merged.pdf --merge

    # Use wildcard pattern
    python png_to_pdf.py "*.png" combined.pdf --merge
"""

import argparse
import glob
import os
import sys
from pathlib import Path

from PIL import Image


def png_to_pdf(input_path: str, output_path: str, dpi: int = 72, quality: int = 95) -> None:
    """Convert a single PNG file to PDF."""
    img = Image.open(input_path)
    
    # Convert to RGB if necessary (PDF doesn't support RGBA, P, etc.)
    if img.mode in ('RGBA', 'LA', 'P'):
        # For images with transparency, composite onto white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        if img.mode in ('RGBA', 'LA'):
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        else:
            img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Save as PDF
    img.save(output_path, 'PDF', resolution=dpi, quality=quality)
    print(f"Converted: {input_path} -> {output_path}")


def merge_pngs_to_pdf(input_paths: list[str], output_path: str, dpi: int = 72, quality: int = 95) -> None:
    """Merge multiple PNG files into a single PDF."""
    if not input_paths:
        print("Error: No input files found.")
        sys.exit(1)
    
    images = []
    for path in input_paths:
        img = Image.open(path)
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            if img.mode in ('RGBA', 'LA'):
                background.paste(img, mask=img.split()[-1])
                img = background
            else:
                img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        images.append(img)
    
    # Save first image and append the rest
    first_image = images[0]
    rest_images = images[1:] if len(images) > 1 else []
    
    first_image.save(
        output_path, 
        'PDF', 
        resolution=dpi, 
        save_all=True, 
        append_images=rest_images,
        quality=quality
    )
    print(f"Merged {len(images)} images into: {output_path}")


def get_png_files(input_pattern: str) -> list[str]:
    """Get list of PNG files from input pattern or directory."""
    path = Path(input_pattern)
    
    # If it's a directory, get all PNG files
    if path.is_dir():
        files = sorted(path.glob('*.png'))
        return [str(f) for f in files]
    
    # If it's a file, return it
    if path.is_file():
        return [str(path)]
    
    # Otherwise treat as glob pattern
    files = sorted(glob.glob(input_pattern))
    png_files = [f for f in files if f.lower().endswith('.png')]
    return png_files


def auto_output_path(input_path: str, is_merge: bool = False) -> str:
    """Generate output PDF path automatically."""
    path = Path(input_path)
    
    if path.is_dir():
        if is_merge:
            return str(path / 'merged.pdf')
        else:
            return str(path)
    else:
        return str(path.with_suffix('.pdf'))


def main():
    parser = argparse.ArgumentParser(
        description='Convert PNG files to PDF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.png
  %(prog)s image.pdf output.pdf
  %(prog)s ./photos/
  %(prog)s ./photos/ merged.pdf --merge
  %(prog)s "*.png" combined.pdf --merge
        """
    )
    parser.add_argument('input', help='Input PNG file, directory, or wildcard pattern')
    parser.add_argument('output', nargs='?', help='Output PDF file (optional)')
    parser.add_argument('--merge', action='store_true', help='Merge multiple PNGs into a single PDF')
    parser.add_argument('--dpi', type=int, default=72, help='DPI for output PDF (default: 72)')
    parser.add_argument('--quality', type=int, default=95, help='Quality for JPEG compression (default: 95)')
    
    args = parser.parse_args()
    
    # Get input files
    input_files = get_png_files(args.input)
    
    if not input_files:
        print(f"Error: No PNG files found matching '{args.input}'")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = auto_output_path(args.input, args.merge)
    
    # Execute conversion
    if args.merge or len(input_files) > 1 and args.output:
        # Merge mode
        merge_pngs_to_pdf(input_files, output_path, dpi=args.dpi, quality=args.quality)
    elif len(input_files) == 1:
        # Single file
        if Path(output_path).is_dir():
            output_path = str(Path(output_path) / (Path(input_files[0]).stem + '.pdf'))
        png_to_pdf(input_files[0], output_path, dpi=args.dpi, quality=args.quality)
    else:
        # Multiple files, separate PDFs
        for file in input_files:
            out = str(Path(file).with_suffix('.pdf'))
            png_to_pdf(file, out, dpi=args.dpi, quality=args.quality)


if __name__ == '__main__':
    main()

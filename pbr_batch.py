#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pbr_batch.py — batch-generate PBR maps from seamless PNG textures.
Outputs: _albedo, _height, _normal, _roughness, _ao, _metallic, optional _ORM.
Preserves seamless tiling via wrap-padding. Can zip results.

Usage examples:
  python3 pbr_batch.py -i ./in -o ./out
  python3 pbr_batch.py -i ./in -o ./out --pack-orm --zip-results
  python3 pbr_batch.py -i ./in -o ./out --pack-orm --zip-results --zip-per-folder
  python3 pbr_batch.py -i ./in -o ./out --normal-strength 4.0 --roughness-contrast 1.3 --ao-strength 1.5

Author: AITEXTURED
License: MIT
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import fnmatch
import zipfile
import logging

import numpy as np
import cv2


# -----------------------------
# Utilities
# -----------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def find_images(root: Path, pattern: str):
    pattern = pattern or "*.png"
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if fnmatch.fnmatch(name, pattern):
                yield Path(dirpath) / name


def read_png(path: Path):
    # Read with alpha if present
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    # Convert 16-bit to 8-bit if needed
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)
    return img


def write_png(path: Path, arr: np.ndarray):
    ensure_dir(path)
    ok = cv2.imwrite(str(path), arr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def to_gray_float01(img_bgr: np.ndarray) -> np.ndarray:
    """BGR to perceptual gray float [0,1]."""
    b, g, r = cv2.split(img_bgr.astype(np.float32) / 255.0)
    gray = 0.0722 * b + 0.7152 * g + 0.2126 * r
    return np.clip(gray, 0.0, 1.0)


def normalize01(x: np.ndarray) -> np.ndarray:
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def gaussian_blur_wrap(img: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_WRAP)


def sobel_wrap(img: np.ndarray, dx: int, dy: int, ksize: int = 3) -> np.ndarray:
    return cv2.Sobel(img, cv2.CV_32F, dx=dx, dy=dy, ksize=ksize, borderType=cv2.BORDER_WRAP)


# -----------------------------
# Map Generators
# -----------------------------
def gen_height_from_albedo(albedo_bgr: np.ndarray) -> np.ndarray:
    h = to_gray_float01(albedo_bgr)
    return (h * 255.0).astype(np.uint8)


def gen_normal_from_height(height_gray_u8: np.ndarray, strength: float = 3.0) -> np.ndarray:
    h = height_gray_u8.astype(np.float32) / 255.0

    # Gradient via Sobel (wrap border)
    gx = sobel_wrap(h, 1, 0, ksize=3)
    gy = sobel_wrap(h, 0, 1, ksize=3)

    # Scale gradients; invert to match classic tangent-space convention if needed
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(h, dtype=np.float32)

    # Normalize
    length = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    nx /= length
    ny /= length
    nz /= length

    # Map [-1,1] -> [0,255] as RGB then convert to BGR for OpenCV write
    nmap_rgb = np.dstack((
        (nx * 0.5 + 0.5),
        (ny * 0.5 + 0.5),
        (nz * 0.5 + 0.5),
    )).astype(np.float32)
    nmap_rgb = np.clip(nmap_rgb * 255.0, 0, 255).astype(np.uint8)
    nmap_bgr = cv2.cvtColor(nmap_rgb, cv2.COLOR_RGB2BGR)
    return nmap_bgr


def gen_roughness_from_local_contrast(albedo_bgr: np.ndarray, contrast_boost: float = 1.2) -> np.ndarray:
    # Work in gray float
    g = to_gray_float01(albedo_bgr)

    # Multi-scale contrast (wrap-safe)
    c1 = np.abs(g - gaussian_blur_wrap(g, ksize=3,  sigma=1.0))
    c2 = np.abs(g - gaussian_blur_wrap(g, ksize=5,  sigma=2.0))
    c3 = np.abs(g - gaussian_blur_wrap(g, ksize=9,  sigma=4.0))

    c = (0.5 * c1 + 0.35 * c2 + 0.15 * c3) * contrast_boost
    rough = normalize01(c)
    return (np.clip(rough, 0.0, 1.0) * 255.0).astype(np.uint8)


def gen_ao_from_height(height_gray_u8: np.ndarray, strength: float = 1.0) -> np.ndarray:
    h = height_gray_u8.astype(np.float32) / 255.0

    b2 = gaussian_blur_wrap(h, ksize=5,  sigma=2.0)
    b4 = gaussian_blur_wrap(h, ksize=9,  sigma=4.0)
    b8 = gaussian_blur_wrap(h, ksize=15, sigma=8.0)

    occl = np.maximum(0.0, ((b2 - h) * 0.5 + (b4 - h) * 0.35 + (b8 - h) * 0.15))
    occl = normalize01(occl)

    # AO = 1 - occlusion * strength
    ao = np.clip(1.0 - occl * strength, 0.0, 1.0)
    return (ao * 255.0).astype(np.uint8)


def gen_metallic_constant(shape, value: float = 0.0) -> np.ndarray:
    value = float(np.clip(value, 0.0, 1.0))
    return np.full(shape[:2], int(round(value * 255.0)), dtype=np.uint8)


def pack_orm(ao_u8: np.ndarray, rough_u8: np.ndarray, metal_u8: np.ndarray) -> np.ndarray:
    # ORM (R=AO, G=Roughness, B=Metallic)
    return cv2.merge([ao_u8, rough_u8, metal_u8])


# -----------------------------
# Processing a single image
# -----------------------------
def process_image(
    in_path: Path,
    out_root: Path,
    rel_root: Path,
    args: argparse.Namespace,
):
    img = read_png(in_path)

    # Preserve albedo (copy original RGB; keep alpha if any)
    if img.ndim == 3 and img.shape[2] == 4:
        albedo_bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    elif img.ndim == 3:
        albedo_bgr = img
        alpha = None
    else:
        albedo_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alpha = None

    stem = in_path.stem
    out_dir = (out_root / rel_root)
    ensure_dir(out_dir / ".__ensure_dir.txt")  # create folders

    # Albedo
    albedo_path = out_dir / f"{stem}_albedo.png"
    if alpha is not None:
        albedo_rgba = cv2.cvtColor(albedo_bgr, cv2.COLOR_BGR2BGRA)
        albedo_rgba[:, :, 3] = alpha
        write_png(albedo_path, albedo_rgba)
    else:
        write_png(albedo_path, albedo_bgr)

    # Height
    height_u8 = gen_height_from_albedo(albedo_bgr)
    if not args.skip_height:
        write_png(out_dir / f"{stem}_height.png", height_u8)

    # Normal
    normal_u8_bgr = gen_normal_from_height(height_u8, strength=args.normal_strength)
    write_png(out_dir / f"{stem}_normal.png", normal_u8_bgr)

    # Roughness (by multi-scale local contrast)
    rough_u8 = gen_roughness_from_local_contrast(albedo_bgr, contrast_boost=args.roughness_contrast)
    if args.keep_glossy:
        # user requests to keep generated value as "gloss", convert to roughness here
        rough_u8 = (255 - rough_u8).astype(np.uint8)
    write_png(out_dir / f"{stem}_roughness.png", rough_u8)

    # AO
    ao_u8 = gen_ao_from_height(height_u8, strength=args.ao_strength)
    write_png(out_dir / f"{stem}_ao.png", ao_u8)

    # Metallic
    metal_u8 = gen_metallic_constant(albedo_bgr, value=args.metallic)
    write_png(out_dir / f"{stem}_metallic.png", metal_u8)

    # ORM
    if args.pack_orm:
        orm = pack_orm(ao_u8, rough_u8, metal_u8)
        write_png(out_dir / f"{stem}_ORM.png", orm)

    logging.info(f"Processed: {in_path} -> {out_dir}")


# -----------------------------
# ZIP helpers
# -----------------------------
def zip_dir(root_dir: Path, zip_path: Path):
    ensure_dir(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in root_dir.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(root_dir)))


def zip_per_folder(out_root: Path):
    for child in out_root.iterdir():
        if child.is_dir():
            zip_path = out_root / f"{child.name}.zip"
            zip_dir(child, zip_path)
            logging.info(f"Created per-folder zip: {zip_path}")
        elif child.is_file():
            zip_path = out_root / f"{child.stem}.zip"
            ensure_dir(zip_path)
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(child, arcname=child.name)
            logging.info(f"Created per-file zip: {zip_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    setup_logging()

    ap = argparse.ArgumentParser(
        description="Batch-generate PBR maps (Height/Normal/Roughness/AO/Metallic, optional ORM) from seamless PNGs. Preserves tiling via wrap-padding."
    )
    ap.add_argument("-i", "--input", required=True, type=Path, help="Input folder (recursively scans)")
    ap.add_argument("-o", "--output", required=True, type=Path, help="Output folder (mirrors structure)")
    ap.add_argument("--glob", default="*.png", help="Filename pattern (default: *.png)")

    ap.add_argument("--normal-strength", type=float, default=3.0, help="Normal intensity (default 3.0)")
    ap.add_argument("--roughness-contrast", type=float, default=1.2, help="Local contrast boost for roughness (default 1.2)")
    ap.add_argument("--ao-strength", type=float, default=1.0, help="AO strength (default 1.0)")
    ap.add_argument("--metallic", type=float, default=0.0, help="Constant metallic value in [0..1] (default 0)")

    ap.add_argument("--pack-orm", action="store_true", help="Export packed ORM (R=AO, G=Roughness, B=Metallic)")
    ap.add_argument("--skip-height", action="store_true", help="Do not export a separate height map")
    ap.add_argument("--keep-glossy", action="store_true", help="Do not invert generated gloss to roughness")

    ap.add_argument("--zip-results", action="store_true", help="Zip the entire output folder")
    ap.add_argument("--zip-name", type=str, default=None, help="Custom name for the main zip (e.g., results.zip)")
    ap.add_argument("--zip-per-folder", action="store_true", help="Also create one zip per immediate subfolder")

    args = ap.parse_args()

    in_root: Path = args.input
    out_root: Path = args.output

    if not in_root.exists() or not in_root.is_dir():
        logging.error(f"Input folder not found or not a directory: {in_root}")
        sys.exit(1)

    all_imgs = list(find_images(in_root, args.glob))
    if not all_imgs:
        logging.warning("No images found with pattern '%s' in %s", args.glob, in_root)
        sys.exit(0)

    for img_path in all_imgs:
        rel = img_path.parent.relative_to(in_root)
        try:
            process_image(img_path, out_root, rel, args)
        except Exception as e:
            logging.exception(f"Error processing {img_path}: {e}")

    if args.zip_per_folder:
        zip_per_folder(out_root)

    if args.zip_results:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = args.zip_name or f"out_{ts}.zip"
        zip_path = out_root.parent / zip_name
        zip_dir(out_root, zip_path)
        logging.info(f"Created main zip: {zip_path}")


if __name__ == "__main__":
    main()

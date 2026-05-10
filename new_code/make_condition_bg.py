#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据原始超声图像与真实结节 mask 生成 Tiger 所需的 condition_BG 图像。

默认规则：
1) 从 JSON 读取原图相对路径（默认字段 filename）。
2) mask 不写入 JSON，而是根据原图相对路径在 --mask_dir 下自动推断：
   - 保持相同相对路径和文件 stem
   - 扩展名可不同，在常见图像扩展名中自动查找
3) 输出到 --output_dir 下，默认镜像原图相对路径，统一保存为 png。
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def read_json_records(json_path: Path) -> List[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON 顶层必须是列表")
    return data


def normalize_rel_path(value: str) -> str:
    return str(value).strip().replace("\\", "/")


def resolve_existing_stem_path(root_dir: Path, rel_filename: str) -> Optional[Path]:
    rel_path = Path(rel_filename)
    stem = rel_path.stem
    parent = root_dir / rel_path.parent
    for suffix in IMAGE_SUFFIXES:
        candidate = parent / f"{stem}{suffix}"
        if candidate.exists():
            return candidate.resolve()
    return None


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_grayscale_image(image_path: Path, size: int) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def load_binary_mask(mask_path: Path, size: int, mask_threshold: int) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法读取 mask: {mask_path}")
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(mask, mask_threshold, 255, cv2.THRESH_BINARY)
    return binary


def dilate_mask(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    if dilate_px <= 0:
        return mask
    kernel_size = max(1, int(dilate_px))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.uint8)
    normalized = (image - min_value) / (max_value - min_value)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def build_sobel_image(gray_image: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    sobel_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    return normalize_uint8(magnitude)


def build_condition_bg(
    gray_image: np.ndarray,
    lesion_mask: np.ndarray,
    mode: str,
    hybrid_gray_weight: float,
) -> np.ndarray:
    sobel_image = build_sobel_image(gray_image)
    gray_norm = normalize_uint8(gray_image)

    if mode == "gray":
        condition = gray_norm
    elif mode == "sobel":
        condition = sobel_image
    else:
        alpha = float(hybrid_gray_weight)
        alpha = min(1.0, max(0.0, alpha))
        beta = 1.0 - alpha
        condition = cv2.addWeighted(gray_norm, alpha, sobel_image, beta, 0)

    condition = condition.copy()
    condition[lesion_mask > 0] = 0
    return condition


def iter_resolved_pairs(
    records: Iterable[dict],
    base_dir: Path,
    mask_dir: Path,
    filename_key: str,
) -> Iterable[Tuple[str, Path, Optional[Path]]]:
    for rec in records:
        if filename_key not in rec:
            yield "", None, None
            continue
        rel_filename = normalize_rel_path(rec[filename_key])
        image_path = (base_dir / rel_filename).resolve()
        mask_path = resolve_existing_stem_path(mask_dir, rel_filename)
        yield rel_filename, image_path, mask_path


def build_output_path(output_dir: Path, rel_filename: str) -> Path:
    rel_path = Path(rel_filename)
    return (output_dir / rel_path.parent / f"{rel_path.stem}.png").resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成 Tiger 所需的 condition_BG 图像")
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--filename_key", type=str, default="filename")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--dilate_px", type=int, default=5)
    parser.add_argument("--mode", type=str, choices=["hybrid", "gray", "sobel"], default="hybrid")
    parser.add_argument("--hybrid_gray_weight", type=float, default=0.5)
    parser.add_argument("--require_exists", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    json_path = Path(args.json_path).resolve()
    base_dir = Path(args.base_dir).resolve()
    mask_dir = Path(args.mask_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not json_path.exists():
        raise FileNotFoundError(f"JSON 不存在: {json_path}")

    records = read_json_records(json_path)

    stats = {
        "total_records": 0,
        "missing_filename_key": 0,
        "missing_image": 0,
        "missing_mask": 0,
        "skipped_existing": 0,
        "generated": 0,
    }

    for rel_filename, image_path, mask_path in iter_resolved_pairs(
        records=records,
        base_dir=base_dir,
        mask_dir=mask_dir,
        filename_key=args.filename_key,
    ):
        stats["total_records"] += 1

        if not rel_filename:
            stats["missing_filename_key"] += 1
            continue

        if not image_path.exists():
            stats["missing_image"] += 1
            if args.require_exists:
                raise FileNotFoundError(f"原图不存在: {image_path}")
            continue

        if mask_path is None or not mask_path.exists():
            stats["missing_mask"] += 1
            if args.require_exists:
                raise FileNotFoundError(f"mask 不存在: {rel_filename}")
            continue

        output_path = build_output_path(output_dir=output_dir, rel_filename=rel_filename)
        if output_path.exists() and not args.overwrite:
            stats["skipped_existing"] += 1
            continue

        gray_image = load_grayscale_image(image_path=image_path, size=args.size)
        lesion_mask = load_binary_mask(mask_path=mask_path, size=args.size, mask_threshold=args.mask_threshold)
        lesion_mask = dilate_mask(lesion_mask, args.dilate_px)
        condition_bg = build_condition_bg(
            gray_image=gray_image,
            lesion_mask=lesion_mask,
            mode=args.mode,
            hybrid_gray_weight=args.hybrid_gray_weight,
        )

        ensure_parent(output_path)
        ok = cv2.imwrite(str(output_path), condition_bg)
        if not ok:
            raise RuntimeError(f"写入失败: {output_path}")
        stats["generated"] += 1

    print("=" * 72)
    print("condition_BG 生成完成")
    print("=" * 72)
    for key, value in stats.items():
        print(f"{key:>24}: {value}")
    print("输出目录:", output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JSON 驱动版 run_tiger_model2

能力：
1) 读取 JSON 标注并构建 train/valid，并可选接入独立 test。
2) 输出 ResNet 目录结构与 CSV（Path,label,subtype）。
3) 可选接入 Tiger 扩散生成，仅对训练集少数类按倍率生成，并并入训练集。
4) 支持自动推断真实结节 mask 与 condition_BG。
5) 支持单阶段生成或近似论文流程的 FG->BG 两阶段生成。
"""

import argparse
import importlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Sample:
    src_path: Path
    rel_filename: str
    raw_label: int
    label: int
    subtype: str
    source_type: str = "original"
    mask_path: Optional[Path] = None
    condition_bg_path: Optional[Path] = None
    condition_fg_path: Optional[Path] = None


def parse_label_map(label_map_str: str) -> Dict[int, Tuple[str, int]]:
    mapping: Dict[int, Tuple[str, int]] = {}
    for item in label_map_str.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"label_map 项格式错误: {item}")
        raw, subtype, out_label = parts
        mapping[int(raw)] = (subtype, int(out_label))
    if not mapping:
        raise ValueError("label_map 不能为空")
    return mapping


def parse_ignore_labels(ignore_labels: str) -> Sequence[int]:
    values = []
    for item in ignore_labels.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def read_json_records(json_path: Path) -> List[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON 顶层必须是列表")
    return data


def resolve_existing_stem_path(root_dir: Optional[Path], rel_filename: str) -> Optional[Path]:
    if root_dir is None:
        return None
    rel_path = Path(rel_filename)
    stem = rel_path.stem
    parent = root_dir / rel_path.parent
    if not parent.exists():
        return None

    stem_lower = stem.lower()
    for candidate in parent.iterdir():
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if candidate.stem.lower() == stem_lower:
            return candidate.resolve()
    return None


def build_samples(
    records: Iterable[dict],
    base_dir: Path,
    filename_key: str,
    label_key: str,
    label_map: Dict[int, Tuple[str, int]],
    ignore_labels: Sequence[int],
    require_exists: bool,
    mask_dir: Optional[Path] = None,
    condition_bg_dir: Optional[Path] = None,
    condition_fg_dir: Optional[Path] = None,
) -> Tuple[List[Sample], Dict[str, int]]:
    stats = {
        "total_records": 0,
        "missing_keys": 0,
        "ignored_label": 0,
        "unknown_label": 0,
        "invalid_image_suffix": 0,
        "missing_file": 0,
        "accepted": 0,
        "resolved_mask": 0,
        "resolved_condition_bg": 0,
        "resolved_condition_fg": 0,
    }
    samples: List[Sample] = []

    for rec in records:
        stats["total_records"] += 1
        if filename_key not in rec or label_key not in rec:
            stats["missing_keys"] += 1
            continue

        rel_filename = str(rec[filename_key]).strip().replace("\\", "/")
        raw_label = rec[label_key]
        if raw_label is None:
            stats["ignored_label"] += 1
            continue

        try:
            raw_label_int = int(raw_label)
        except (TypeError, ValueError):
            stats["unknown_label"] += 1
            continue

        if raw_label_int in ignore_labels:
            stats["ignored_label"] += 1
            continue
        if raw_label_int not in label_map:
            stats["unknown_label"] += 1
            continue

        src_path = (base_dir / rel_filename).resolve()
        if src_path.suffix.lower() not in IMAGE_SUFFIXES:
            stats["invalid_image_suffix"] += 1
            continue
        if require_exists and not src_path.exists():
            stats["missing_file"] += 1
            continue

        mask_path = resolve_existing_stem_path(mask_dir, rel_filename)
        condition_bg_path = resolve_existing_stem_path(condition_bg_dir, rel_filename)
        condition_fg_path = resolve_existing_stem_path(condition_fg_dir, rel_filename)
        if mask_path is not None:
            stats["resolved_mask"] += 1
        if condition_bg_path is not None:
            stats["resolved_condition_bg"] += 1
        if condition_fg_path is not None:
            stats["resolved_condition_fg"] += 1

        subtype, out_label = label_map[raw_label_int]
        samples.append(
            Sample(
                src_path=src_path,
                rel_filename=rel_filename,
                raw_label=raw_label_int,
                label=out_label,
                subtype=subtype,
                mask_path=mask_path,
                condition_bg_path=condition_bg_path,
                condition_fg_path=condition_fg_path,
            )
        )

    stats["accepted"] = len(samples)
    return samples, stats


def split_samples(
    samples: List[Sample],
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> Dict[str, List[Sample]]:
    if not samples:
        raise ValueError("无可用样本，无法切分")
    total = train_ratio + valid_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/valid 比例之和必须为 1.0")

    y = [s.label for s in samples]
    train, valid = train_test_split(
        samples,
        test_size=valid_ratio,
        random_state=seed,
        stratify=y,
    )
    return {"train": train, "valid": valid}


def reset_output_dirs(resnet_root: Path, include_test: bool = False) -> None:
    splits = ["train", "valid"] + (["test"] if include_test else [])
    for split in splits:
        for label in ("0", "1"):
            d = resnet_root / split / label
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)


def safe_dst_name(sample: Sample, used_names: Dict[str, int]) -> str:
    stem = sample.rel_filename.replace("/", "__").replace("\\", "__")
    base = stem
    suffix = sample.src_path.suffix.lower()
    if not base.lower().endswith(suffix):
        base = f"{base}{suffix}"

    count = used_names.get(base, 0)
    used_names[base] = count + 1
    if count == 0:
        return base
    pure = base[: -len(suffix)] if suffix else base
    return f"{pure}__dup{count}{suffix}"


def create_fg_prompt(subtype: str) -> str:
    subtype_upper = subtype.upper()
    if "PTC" in subtype_upper:
        return "papillary thyroid carcinoma, malignant nodule, thyroid ultrasound lesion"
    if "FTC" in subtype_upper:
        return "follicular thyroid carcinoma, malignant nodule, thyroid ultrasound lesion"
    return "thyroid malignant nodule, ultrasound lesion"


def create_bg_prompt() -> str:
    return "thyroid ultrasound background, grayscale texture, soft tissue detail"


def build_center_mask(image_size: Tuple[int, int]) -> Image.Image:
    w, h = image_size
    yy, xx = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    radius = max(8, min(w, h) // 3)
    dist = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[dist <= radius * radius] = 255
    return Image.fromarray(mask, mode="L")


def load_tiger_pipeline(pretrain_model_path: str, controlnet_bg_path: str, dtype: str):
    try:
        import torch

        diffusers = importlib.import_module("diffusers")
        DDIMScheduler = getattr(diffusers, "DDIMScheduler")
        ControlNetModel = getattr(diffusers, "ControlNetModel")
        StableDiffusionControlNetInpaintPipeline = getattr(
            diffusers, "StableDiffusionControlNetInpaintPipeline"
        )
    except Exception as exc:
        raise RuntimeError(
            "加载 Tiger 依赖失败，请安装 diffusers/torch 等依赖"
        ) from exc

    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    controlnet_bg = ControlNetModel.from_pretrained(controlnet_bg_path, torch_dtype=torch_dtype)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        pretrain_model_path,
        controlnet=controlnet_bg,
        torch_dtype=torch_dtype,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    return pipe


def resolve_multiplier_for_subtype(subtype: str, class_multipliers: Dict[str, int]) -> int:
    subtype_upper = subtype.upper()
    if "PTC" in subtype_upper:
        return max(0, int(class_multipliers.get("PTC", 0)))
    if "FTC" in subtype_upper:
        return max(0, int(class_multipliers.get("FTC", 0)))
    return max(0, int(class_multipliers.get(subtype_upper, 0)))


def load_mask_image(mask_path: Optional[Path], image_size: Tuple[int, int]) -> Image.Image:
    if mask_path is None or not mask_path.exists():
        return build_center_mask(image_size)
    return Image.open(mask_path).convert("L").resize(image_size, Image.Resampling.NEAREST)


def load_condition_bg_image(
    condition_bg_path: Optional[Path],
    fallback_image: Image.Image,
    image_size: Tuple[int, int],
) -> Image.Image:
    if condition_bg_path is None or not condition_bg_path.exists():
        return fallback_image
    return Image.open(condition_bg_path).convert("RGB").resize(image_size, Image.Resampling.BILINEAR)


def load_condition_fg_image(
    condition_fg_path: Optional[Path],
    mask_path: Optional[Path],
    image_size: Tuple[int, int],
) -> Image.Image:
    if condition_fg_path is not None and condition_fg_path.exists():
        return Image.open(condition_fg_path).convert("L").resize(image_size, Image.Resampling.NEAREST)
    return load_mask_image(mask_path, image_size)


def build_background_mask(mask_image: Image.Image, dilate_px: int) -> Image.Image:
    mask_array = np.array(mask_image.convert("L"), dtype=np.uint8)
    if dilate_px > 0:
        kernel_size = max(1, int(dilate_px))
        if kernel_size % 2 == 0:
            kernel_size += 1
        try:
            import cv2

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask_array = cv2.dilate(mask_array, kernel, iterations=1)
        except Exception:
            pass
    bg_mask = 255 - mask_array
    return Image.fromarray(bg_mask, mode="L")


def run_single_stage_generation(
    pipe,
    src_image: Image.Image,
    mask_image: Image.Image,
    control_image: Image.Image,
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed_i: int,
):
    import torch

    return pipe(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed_i),
        eta=1.0,
        controlnet_conditioning_scale=1.0,
        image=src_image,
        mask_image=mask_image,
        control_image=control_image,
    ).images[0]


def run_two_stage_generation(
    pipe,
    src_image: Image.Image,
    fg_mask_image: Image.Image,
    fg_control_image: Image.Image,
    bg_mask_image: Image.Image,
    bg_control_image: Image.Image,
    fg_prompt: str,
    bg_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed_i: int,
):
    fg_image = run_single_stage_generation(
        pipe=pipe,
        src_image=src_image,
        mask_image=fg_mask_image,
        control_image=fg_control_image,
        prompt=fg_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed_i=seed_i,
    )
    bg_image = run_single_stage_generation(
        pipe=pipe,
        src_image=fg_image.convert("RGB"),
        mask_image=bg_mask_image,
        control_image=bg_control_image,
        prompt=bg_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed_i=seed_i + 1,
    )
    return bg_image


def augment_train_with_tiger(
    train_samples: List[Sample],
    output_root: Path,
    pretrain_model_path: str,
    controlnet_bg_path: str,
    class_multipliers: Dict[str, int],
    num_inference_steps: int,
    guidance_scale: float,
    dtype: str,
    seed: int,
    use_two_stage_generation: bool,
    bg_mask_dilate_px: int,
) -> Tuple[List[Sample], Dict[str, int]]:
    class_counts: Dict[int, int] = {}
    planned_per_label: Dict[int, int] = {}
    for sample in train_samples:
        class_counts[sample.label] = class_counts.get(sample.label, 0) + 1
        planned_per_label[sample.label] = planned_per_label.get(sample.label, 0) + resolve_multiplier_for_subtype(
            sample.subtype, class_multipliers
        )

    planned_generated = sum(planned_per_label.values())
    if planned_generated <= 0:
        return [], {
            "planned_generated": 0,
            "actual_generated": 0,
            "existing_generated": 0,
            "newly_generated": 0,
            "class_counts_before": str(class_counts),
            "planned_per_label": str(planned_per_label),
        }

    generated_root_base = output_root / "generated_images" / "train"
    existing_by_label: Dict[int, List[Path]] = {}
    for label in planned_per_label:
        label_dir = generated_root_base / str(label)
        if not label_dir.exists():
            existing_by_label[label] = []
            continue
        files = sorted(
            p for p in label_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        existing_by_label[label] = files

    existing_generated = sum(len(paths) for paths in existing_by_label.values())
    enough_existing = all(
        len(existing_by_label.get(label, [])) >= planned_per_label.get(label, 0)
        for label in planned_per_label
    )

    generated_samples: List[Sample] = []
    for label, planned_count in planned_per_label.items():
        for path in existing_by_label.get(label, [])[:planned_count]:
            generated_samples.append(
                Sample(
                    src_path=path,
                    rel_filename=path.relative_to(output_root).as_posix(),
                    raw_label=label,
                    label=label,
                    subtype=f"generated_label_{label}",
                    source_type="generated",
                )
            )

    if enough_existing:
        class_counts_after = dict(class_counts)
        for g in generated_samples:
            class_counts_after[g.label] = class_counts_after.get(g.label, 0) + 1
        report = {
            "planned_generated": planned_generated,
            "actual_generated": len(generated_samples),
            "existing_generated": existing_generated,
            "newly_generated": 0,
            "class_counts_before": str(class_counts),
            "class_counts_after": str(class_counts_after),
            "planned_per_label": str(planned_per_label),
            "existing_per_label": str({label: len(paths) for label, paths in existing_by_label.items()}),
            "ptc_generate_per_image": class_multipliers.get("PTC", 0),
            "ftc_generate_per_image": class_multipliers.get("FTC", 0),
        }
        return generated_samples, report

    pipe = load_tiger_pipeline(pretrain_model_path, controlnet_bg_path, dtype)
    rng = random.Random(seed)
    generated_now = 0
    generated_index_per_label = {label: len(paths) for label, paths in existing_by_label.items()}

    for sample in train_samples:
        per_image = resolve_multiplier_for_subtype(sample.subtype, class_multipliers)
        if per_image <= 0:
            continue

        label = sample.label
        label_target = planned_per_label.get(label, 0)
        label_current = generated_index_per_label.get(label, 0)
        if label_current >= label_target:
            continue

        generated_root = output_root / "generated_images" / "train" / str(label)
        generated_root.mkdir(parents=True, exist_ok=True)
        fg_prompt = create_fg_prompt(sample.subtype)
        bg_prompt = create_bg_prompt()

        needed_for_sample = min(per_image, label_target - label_current)
        generated_success = 0
        for _ in range(needed_for_sample):
            seed_i = rng.randint(1, 10**9)
            image_size = (512, 512)
            src_image = Image.open(sample.src_path).convert("RGB").resize(image_size, Image.Resampling.BILINEAR)
            fg_mask_image = load_mask_image(sample.mask_path, image_size)
            fg_control_image = load_condition_fg_image(sample.condition_fg_path, sample.mask_path, image_size).convert("RGB")
            bg_mask_image = build_background_mask(fg_mask_image, bg_mask_dilate_px)
            bg_control_image = load_condition_bg_image(sample.condition_bg_path, src_image, image_size)

            try:
                if use_two_stage_generation:
                    image = run_two_stage_generation(
                        pipe=pipe,
                        src_image=src_image,
                        fg_mask_image=fg_mask_image,
                        fg_control_image=fg_control_image,
                        bg_mask_image=bg_mask_image,
                        bg_control_image=bg_control_image,
                        fg_prompt=fg_prompt,
                        bg_prompt=bg_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        seed_i=seed_i,
                    )
                else:
                    image = run_single_stage_generation(
                        pipe=pipe,
                        src_image=src_image,
                        mask_image=fg_mask_image,
                        control_image=bg_control_image,
                        prompt=fg_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        seed_i=seed_i,
                    )
            except Exception as exc:
                print(f"⚠️  生成失败，跳过: {sample.src_path.name} ({exc})")
                continue

            safe_name = sample.rel_filename.replace("/", "__").replace("\\", "__")
            out_name = f"generated_{sample.subtype}_{label_current + generated_success}_{safe_name}_{seed_i}.png"
            out_path = generated_root / out_name
            image.save(out_path)

            generated_samples.append(
                Sample(
                    src_path=out_path,
                    rel_filename=f"generated_images/train/{label}/{out_name}",
                    raw_label=sample.raw_label,
                    label=label,
                    subtype=sample.subtype,
                    source_type="generated",
                )
            )
            generated_now += 1
            generated_success += 1

        generated_index_per_label[label] = generated_index_per_label.get(label, 0) + generated_success

    class_counts_after = dict(class_counts)
    for g in generated_samples:
        class_counts_after[g.label] = class_counts_after.get(g.label, 0) + 1

    report = {
        "planned_generated": planned_generated,
        "actual_generated": len(generated_samples),
        "existing_generated": existing_generated,
        "newly_generated": generated_now,
        "class_counts_before": str(class_counts),
        "class_counts_after": str(class_counts_after),
        "planned_per_label": str(planned_per_label),
        "existing_per_label": str({label: len(paths) for label, paths in existing_by_label.items()}),
        "final_per_label": str(generated_index_per_label),
        "ptc_generate_per_image": class_multipliers.get("PTC", 0),
        "ftc_generate_per_image": class_multipliers.get("FTC", 0),
        "use_two_stage_generation": use_two_stage_generation,
        "bg_mask_dilate_px": bg_mask_dilate_px,
    }
    return generated_samples, report


def copy_split_to_resnet(
    split_name: str,
    split_samples_data: List[Sample],
    output_root: Path,
    resnet_root: Path,
) -> pd.DataFrame:
    rows = []
    used_names: Dict[str, int] = {}

    for sample in split_samples_data:
        label_dir = resnet_root / split_name / str(sample.label)
        label_dir.mkdir(parents=True, exist_ok=True)

        dst_name = safe_dst_name(sample, used_names)
        dst_path = label_dir / dst_name
        shutil.copy2(sample.src_path, dst_path)

        rel_path = dst_path.relative_to(output_root).as_posix()
        rows.append(
            {
                "Path": rel_path,
                "label": int(sample.label),
                "subtype": sample.subtype,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def save_csvs(output_root: Path, dfs: Dict[str, pd.DataFrame], prefix: str) -> None:
    csv_dir = output_root / "CSV"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for split_name, df in dfs.items():
        csv_path = csv_dir / f"{prefix}_{split_name}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")


def print_stats(
    stats: Dict[str, int],
    samples: List[Sample],
    splits: Dict[str, List[Sample]],
    tiger_report: Dict[str, object],
) -> None:
    print("=" * 72)
    print("JSON 解析与筛选统计")
    print("=" * 72)
    for k, v in stats.items():
        print(f"{k:>24}: {v}")

    label_count: Dict[int, int] = {}
    subtype_count: Dict[str, int] = {}
    for s in samples:
        label_count[s.label] = label_count.get(s.label, 0) + 1
        subtype_count[s.subtype] = subtype_count.get(s.subtype, 0) + 1

    print("-" * 72)
    print(f"有效样本标签分布: {label_count}")
    print(f"有效样本亚型分布: {subtype_count}")
    print("-" * 72)
    for split_name in ("train", "valid", "test"):
        split_samples_data = splits.get(split_name, [])
        if not split_samples_data:
            continue
        split_labels: Dict[int, int] = {}
        generated_count = 0
        for s in split_samples_data:
            split_labels[s.label] = split_labels.get(s.label, 0) + 1
            if s.source_type == "generated":
                generated_count += 1
        print(
            f"{split_name:>24}: {len(split_samples_data)}  {split_labels}  generated={generated_count}"
        )
    if tiger_report:
        print("-" * 72)
        print("Tiger 增强统计:")
        for k, v in tiger_report.items():
            print(f"{k:>24}: {v}")
    print("=" * 72)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JSON 驱动版 run_tiger_model2")

    parser.add_argument("--json_path", type=str, default="./new_code/train_labels.json")
    parser.add_argument("--test_json_path", type=str, default=None)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--filename_key", type=str, default="filename")
    parser.add_argument("--label_key", type=str, default="FTCPTC")
    parser.add_argument("--ignore_labels", type=str, default="-1")
    parser.add_argument("--label_map", type=str, default="0:PTC:0,1:FTC:1")

    parser.add_argument("--output_root", type=str, default="./dataset_json")
    parser.add_argument("--csv_prefix", type=str, default="PTC_vs_FTC")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require_exists", action="store_true")
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--condition_bg_dir", type=str, default=None)
    parser.add_argument("--condition_fg_dir", type=str, default=None)

    parser.add_argument("--ptc_generate_per_image", type=int, default=0)
    parser.add_argument("--ftc_generate_per_image", type=int, default=0)
    parser.add_argument("--pretrain_model_path", type=str, default="./model/pretrain")
    parser.add_argument(
        "--controlnet_bg_path",
        type=str,
        default="./model/fine-train-model/controlnet_bg",
    )
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=0.02)
    parser.add_argument("--tiger_dtype", type=str, choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--cap_majority_ratio",
        type=float,
        default=0.0,
        help="若大于 0，则不使用 Tiger 生成，并将训练集中多数类数量限制为最少类的该倍数",
    )
    parser.add_argument(
        "--use_two_stage_generation",
        action="store_true",
        help="启用近似论文流程的 FG->BG 两阶段生成",
    )
    parser.add_argument(
        "--bg_mask_dilate_px",
        type=int,
        default=5,
        help="构造背景 mask 时对前景 mask 的膨胀像素",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    random.seed(args.seed)

    json_path = Path(args.json_path).resolve()
    base_dir = Path(args.base_dir).resolve()
    output_root = Path(args.output_root).resolve()
    mask_dir = Path(args.mask_dir).resolve() if args.mask_dir else None
    condition_bg_dir = Path(args.condition_bg_dir).resolve() if args.condition_bg_dir else None
    condition_fg_dir = Path(args.condition_fg_dir).resolve() if args.condition_fg_dir else None
    resnet_root = output_root / "Resnet_training_data"

    if not json_path.exists():
        raise FileNotFoundError(f"JSON 不存在: {json_path}")

    records = read_json_records(json_path)
    label_map = parse_label_map(args.label_map)
    ignore_labels = parse_ignore_labels(args.ignore_labels)
    samples, stats = build_samples(
        records=records,
        base_dir=base_dir,
        filename_key=args.filename_key,
        label_key=args.label_key,
        label_map=label_map,
        ignore_labels=ignore_labels,
        require_exists=args.require_exists,
        mask_dir=mask_dir,
        condition_bg_dir=condition_bg_dir,
        condition_fg_dir=condition_fg_dir,
    )
    if len(samples) < 2:
        raise ValueError("有效样本数量不足，无法构建训练/验证集")

    splits = split_samples(
        samples=samples,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )

    if args.test_json_path:
        test_json_path = Path(args.test_json_path).resolve()
        if not test_json_path.exists():
            raise FileNotFoundError(f"测试集 JSON 不存在: {test_json_path}")
        test_records = read_json_records(test_json_path)
        test_samples, test_stats = build_samples(
            records=test_records,
            base_dir=base_dir,
            filename_key=args.filename_key,
            label_key=args.label_key,
            label_map=label_map,
            ignore_labels=ignore_labels,
            require_exists=args.require_exists,
            mask_dir=mask_dir,
            condition_bg_dir=condition_bg_dir,
            condition_fg_dir=condition_fg_dir,
        )
        if len(test_samples) == 0:
            raise ValueError("测试集无有效样本，无法构建 test 集")
        splits["test"] = test_samples
        stats["test_total_records"] = test_stats["total_records"]
        stats["test_missing_keys"] = test_stats["missing_keys"]
        stats["test_ignored_label"] = test_stats["ignored_label"]
        stats["test_unknown_label"] = test_stats["unknown_label"]
        stats["test_invalid_image_suffix"] = test_stats["invalid_image_suffix"]
        stats["test_missing_file"] = test_stats["missing_file"]
        stats["test_accepted"] = test_stats["accepted"]

    tiger_report: Dict[str, object] = {}

    if args.cap_majority_ratio > 0:
        from collections import defaultdict

        counts: Dict[int, int] = {}
        for s in splits["train"]:
            counts[s.label] = counts.get(s.label, 0) + 1

        if counts:
            min_count = min(counts.values())
            cap = max(1, int(min_count * args.cap_majority_ratio))
            rng = random.Random(args.seed)

            label_to_samples: Dict[int, List[Sample]] = defaultdict(list)
            for s in splits["train"]:
                label_to_samples[s.label].append(s)

            new_train: List[Sample] = []
            counts_after: Dict[int, int] = {}
            for label, lst in label_to_samples.items():
                if len(lst) > cap:
                    rng.shuffle(lst)
                    kept = lst[:cap]
                else:
                    kept = lst
                new_train.extend(kept)
                counts_after[label] = len(kept)

            tiger_report = {
                "cap_majority_applied": True,
                "cap_majority_ratio": args.cap_majority_ratio,
                "cap_per_label": cap,
                "counts_before": str(counts),
                "counts_after": str(counts_after),
            }
            splits["train"] = new_train
    else:
        if args.ptc_generate_per_image > 0 or args.ftc_generate_per_image > 0:
            generated_samples, tiger_report = augment_train_with_tiger(
                train_samples=splits["train"],
                output_root=output_root,
                pretrain_model_path=args.pretrain_model_path,
                controlnet_bg_path=args.controlnet_bg_path,
                class_multipliers={
                    "PTC": args.ptc_generate_per_image,
                    "FTC": args.ftc_generate_per_image,
                },
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                dtype=args.tiger_dtype,
                seed=args.seed,
                use_two_stage_generation=args.use_two_stage_generation,
                bg_mask_dilate_px=args.bg_mask_dilate_px,
            )
            splits["train"].extend(generated_samples)

    output_root.mkdir(parents=True, exist_ok=True)
    reset_output_dirs(resnet_root, include_test="test" in splits)

    dfs: Dict[str, pd.DataFrame] = {}
    for split_name in ("train", "valid", "test"):
        if split_name not in splits:
            continue
        dfs[split_name] = copy_split_to_resnet(
            split_name=split_name,
            split_samples_data=splits[split_name],
            output_root=output_root,
            resnet_root=resnet_root,
        )

    save_csvs(output_root=output_root, dfs=dfs, prefix=args.csv_prefix)
    print_stats(stats=stats, samples=samples, splits=splits, tiger_report=tiger_report)

    print("输出目录:", output_root)
    print("ResNet 数据目录:", resnet_root)
    print("CSV 目录:", output_root / "CSV")
    print("完成。")


if __name__ == "__main__":
    main()

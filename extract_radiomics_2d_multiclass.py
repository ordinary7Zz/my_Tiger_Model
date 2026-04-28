"""
多类 2D pyradiomics 特征提取：按「类别配置列表」遍历多个 image/mask 目录。
不使用 argparse，请在下方 CLASS_CONFIGS 与全局选项中直接修改路径与参数。
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# 在此处配置：每个类别一条记录
# label: 整数类别编号，写入 CSV 的 label 列（需与训练脚本一致）
# name:  可读名称，写入 CSV 的 subtype 列
# image_dir / mask_dir: 该类图像与掩码目录；掩码命名规则与原脚本相同（与图像 basename 对应，可选 mask 后缀）
# ---------------------------------------------------------------------------
CLASS_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "乳头状癌",
        "label": 1,
        "image_dir": "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Subtype_4cls/cropped/1",
        "mask_dir": "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Subtype_4cls/predictions/subtype_1_predictions",
    },
    {
        "name": "滤泡状癌",
        "label": 2,
        "image_dir": "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Subtype_4cls/cropped/2",
        "mask_dir": "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Subtype_4cls/predictions/subtype_2_predictions",
    },
    {
        "name": "髓样癌",
        "label": 3,
        "image_dir": "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Subtype_4cls/cropped/3",
        "mask_dir": "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Subtype_4cls/predictions/subtype_3_predictions",
    },
    {
        "name": "未分化癌",
        "label": 4,
        "image_dir": "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Subtype_4cls/cropped/4",
        "mask_dir": "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Subtype_4cls/predictions/subtype_4_predictions",
    },
]

OUTPUT_CSV = "./output/radiomics_features_4cls.csv"
PARAMS_YAML = os.path.join(os.path.dirname(__file__), "radiomics_2d.yaml")

MASK_THRESHOLD = 0
MASK_SUFFIX = ""
SPACING_X = 1.0
SPACING_Y = 1.0
SKIP_FAIL = True
LIMIT_PER_CLASS: Optional[int] = None

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG")


def _list_images(image_dir: str) -> List[str]:
    files = []
    for name in os.listdir(image_dir):
        if name.endswith(_IMAGE_EXTS):
            files.append(os.path.join(image_dir, name))
    return sorted(files)


def _read_gray(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape={arr.shape} ({path})")
    return arr


def _read_mask(path: str, threshold: int) -> np.ndarray:
    m = Image.open(path).convert("L")
    arr = np.asarray(m)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={arr.shape} ({path})")
    return (arr > threshold).astype(np.uint8)


def _find_mask(mask_dir: str, image_path: str, mask_suffix: str) -> str:
    base = os.path.splitext(os.path.basename(image_path))[0]
    cand_base = base + (mask_suffix or "")

    for ext in _IMAGE_EXTS:
        cand = os.path.join(mask_dir, cand_base + ext)
        if os.path.exists(cand):
            return cand

    raise FileNotFoundError(
        f"Mask not found for image '{image_path}'. Looked for '{cand_base}'+({', '.join(_IMAGE_EXTS)}) in {mask_dir}"
    )


def _to_sitk(image: np.ndarray, mask: np.ndarray, spacing: Tuple[float, float]):
    import SimpleITK as sitk

    img_sitk = sitk.GetImageFromArray(image.astype(np.float32))
    msk_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))

    img_sitk.SetSpacing(spacing)
    msk_sitk.SetSpacing(spacing)

    return img_sitk, msk_sitk


def _extract_one(
    extractor,
    image_path: str,
    mask_path: str,
    label_value: int,
    subtype_name: str,
    mask_threshold: int,
    spacing: Tuple[float, float],
) -> Dict[str, object]:
    image = _read_gray(image_path)
    mask = _read_mask(mask_path, threshold=mask_threshold)

    if image.shape != mask.shape:
        raise ValueError(f"Image/mask size mismatch: image={image.shape}, mask={mask.shape}")

    if int(mask.sum()) == 0:
        raise ValueError("Empty mask (no foreground pixels)")

    img_sitk, msk_sitk = _to_sitk(image, mask, spacing=spacing)

    result = extractor.execute(img_sitk, msk_sitk, label=1)

    feats = {k: v for k, v in result.items() if not str(k).startswith("diagnostics_")}
    feats["filename"] = os.path.basename(image_path)
    feats["label"] = int(label_value)
    feats["subtype"] = subtype_name
    return feats


def _validate_configs(configs: List[Dict[str, Any]]) -> None:
    if not configs:
        raise ValueError("CLASS_CONFIGS 不能为空")
    labels = [c["label"] for c in configs]
    if len(labels) != len(set(labels)):
        raise ValueError(f"CLASS_CONFIGS 中存在重复的 label: {labels}")
    for i, c in enumerate(configs):
        for key in ("name", "label", "image_dir", "mask_dir"):
            if key not in c:
                raise ValueError(f"CLASS_CONFIGS[{i}] 缺少键 '{key}'")
        if not os.path.isdir(c["image_dir"]):
            raise FileNotFoundError(f"CLASS_CONFIGS[{i}] image_dir 不存在: {c['image_dir']}")
        if not os.path.isdir(c["mask_dir"]):
            raise FileNotFoundError(f"CLASS_CONFIGS[{i}] mask_dir 不存在: {c['mask_dir']}")


def main() -> None:
    _validate_configs(CLASS_CONFIGS)

    from radiomics import featureextractor

    extractor = featureextractor.RadiomicsFeatureExtractor(PARAMS_YAML)

    rows: List[Dict[str, object]] = []
    failures = 0
    counts: Dict[str, int] = {}

    for cfg in CLASS_CONFIGS:
        name = str(cfg["name"])
        label = int(cfg["label"])
        image_dir = cfg["image_dir"]
        mask_dir = cfg["mask_dir"]

        images = _list_images(image_dir)
        if LIMIT_PER_CLASS is not None:
            images = images[:LIMIT_PER_CLASS]

        counts[name] = len(images)
        print(f"Processing {len(images)} images for [{name}] (label={label})...")

        for idx, img_path in enumerate(images):
            fname = os.path.basename(img_path)
            try:
                mask_path = _find_mask(mask_dir, img_path, mask_suffix=MASK_SUFFIX)
                feats = _extract_one(
                    extractor,
                    image_path=img_path,
                    mask_path=mask_path,
                    label_value=label,
                    subtype_name=name,
                    mask_threshold=MASK_THRESHOLD,
                    spacing=(SPACING_X, SPACING_Y),
                )
                rows.append(feats)
            except Exception as e:
                failures += 1
                msg = f"[{name}-{idx}] failed: {fname} err={type(e).__name__}: {e}"
                if SKIP_FAIL:
                    print(msg)
                    continue
                raise RuntimeError(msg) from e

    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(os.path.abspath(OUTPUT_CSV))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    per_class = ", ".join(f"{k}={counts[k]}" for k in counts)
    print(
        f"Done. per_class_images=({per_class}) "
        f"extracted={len(df)} failures={failures} saved={OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()

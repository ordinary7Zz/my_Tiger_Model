# Tiger-Model：run_tiger_model2.py 使用说明（新版 JSON 流程）

## 适用场景

- 希望在现有 JSON 分类流程上，更接近 Tiger 论文的 FG/BG 条件生成逻辑
- 支持自动推断真实 mask、`condition_BG`、`condition_FG`
- 支持单阶段生成与 `FG->BG` 两阶段生成

说明：

- 对应脚本：`new_code/run_tiger_model2.py`
- 这是在旧版 `run_tiger_model.py` 基础上的增强版

---

## 输入约定

1. JSON 顶层是 `list`，元素至少包含：
   - `filename`：图像相对路径（如 `2016/xxx/a.jpg`）
   - `FTCPTC`：标签（默认 `0/1`，`-1` 表示忽略）
2. 自动推断文件：
   - `--mask_dir` 下的真实结节 mask
   - `--condition_bg_dir` 下的 `condition_BG`
   - `--condition_fg_dir` 下的 `condition_FG`（可选）
3. 路径规则：
   - 与原图保持相同相对路径、相同文件 stem
   - 扩展名可不同（`.png` / `.jpg` / `.bmp` / ...）
4. 回退逻辑：
   - 缺 `condition_FG` -> 回退到真实结节 mask
   - 缺 `condition_BG` -> 回退到原图 control
   - 缺 mask -> 回退到中心圆 mask

---

## 步骤 A：推荐先生成 condition_BG

### Ubuntu 小范围验证示例

```bash
python new_code/make_condition_bg.py \
    --json_path ./new_code/json/train_labels_small.json \
    --base_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped \
    --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_masks \
    --output_dir ./dataset_json_debug/condition_BG \
    --filename_key filename \
    --size 512 \
    --mode hybrid \
    --dilate_px 5 \
    --require_exists \
    --overwrite
```

### Ubuntu 全量示例

```bash
python new_code/make_condition_bg.py \
    --json_path ./new_code/json/train_labels.json \
    --base_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped \
    --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_masks \
    --output_dir ./dataset_json/condition_BG \
    --filename_key filename \
    --size 512 \
    --mode hybrid \
    --dilate_px 5 \
    --require_exists
```

---

## 步骤 B：运行 JSON 版 run_tiger_model2（新版）

说明：

- 默认：单阶段生成（更稳、改动更小）
- `--use_two_stage_generation`：启用 `FG->BG` 两阶段生成
- `condition_FG` 可选；不提供时默认回退到真实结节 mask

### 小范围验证示例 A：单阶段生成（推荐先跑）

```bash
python new_code/run_tiger_model2.py \
    --json_path ./new_code/json/train_labels_small.json \
    --test_json_path ./new_code/json/test_labels.json \
    --base_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped \
    --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_masks \
    --condition_bg_dir ./dataset_json_debug/condition_BG \
    --output_root ./dataset_json_debug \
    --csv_prefix PTC_vs_FTC \
    --train_ratio 0.9 \
    --valid_ratio 0.1 \
    --seed 42 \
    --ptc_generate_per_image 0 \
    --ftc_generate_per_image 2 \
    --pretrain_model_path ./model/pretrain \
    --controlnet_bg_path ./model/fine-train-model/controlnet_bg \
    --require_exists
```

### 小范围验证示例 B：两阶段生成（更接近论文流程）

```bash
python new_code/run_tiger_model2.py \
    --json_path ./new_code/json/train_labels_small.json \
    --test_json_path ./new_code/json/test_labels.json \
    --base_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped \
    --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_masks \
    --condition_bg_dir ./dataset_json_debug/condition_BG \
    --output_root ./dataset_json_debug_two_stage \
    --csv_prefix PTC_vs_FTC \
    --train_ratio 0.9 \
    --valid_ratio 0.1 \
    --seed 42 \
    --ptc_generate_per_image 0 \
    --ftc_generate_per_image 2 \
    --pretrain_model_path ./model/pretrain \
    --controlnet_bg_path ./model/fine-train-model/controlnet_bg \
    --use_two_stage_generation \
    --bg_mask_dilate_px 5 \
    --require_exists
```

### 全量运行示例 A：单阶段生成（推荐默认版本）

```bash
python new_code/run_tiger_model2.py \
    --json_path ./new_code/json/train_labels.json \
    --test_json_path ./new_code/json/test_labels.json \
    --base_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped \
    --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_masks \
    --condition_bg_dir ./dataset_json/condition_BG \
    --output_root ./dataset_json_v2 \
    --csv_prefix PTC_vs_FTC \
    --train_ratio 0.9 \
    --valid_ratio 0.1 \
    --seed 42 \
    --ptc_generate_per_image 0 \
    --ftc_generate_per_image 4 \
    --pretrain_model_path ./model/pretrain \
    --controlnet_bg_path ./model/fine-train-model/controlnet_bg \
    --require_exists
```

### 全量运行示例 B：两阶段生成（近似论文流程）

```bash
python new_code/run_tiger_model2.py \
    --json_path ./new_code/json/train_labels.json \
    --test_json_path ./new_code/json/test_labels.json \
    --base_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped \
    --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_masks \
    --condition_bg_dir ./dataset_json/condition_BG \
    --output_root ./dataset_json_v2_two_stage \
    --csv_prefix PTC_vs_FTC \
    --train_ratio 0.9 \
    --valid_ratio 0.1 \
    --seed 42 \
    --ptc_generate_per_image 0 \
    --ftc_generate_per_image 4 \
    --pretrain_model_path ./model/pretrain \
    --controlnet_bg_path ./model/fine-train-model/controlnet_bg \
    --use_two_stage_generation \
    --bg_mask_dilate_px 5 \
    --require_exists
```

### 可选：如果你后续生成了 condition_FG，可以额外传入

```bash
python new_code/run_tiger_model2.py \
    ... \
    --condition_fg_dir ./dataset_json/condition_FG \
    --use_two_stage_generation \
    ...
```

---

## 训练 ResNet 二分类

注意 `imagepath` 要对齐上面的 `output_root`。

```bash
python Resent/main_train.py \
    --modelname Thyroid_PTC_vs_FTC_json \
    --architecture resnet \
    --imagepath ./dataset_json_v2 \
    --train_data PTC_vs_FTC_train \
    --valid_data PTC_vs_FTC_valid \
    --test_data PTC_vs_FTC_test \
    --learning_rate 1e-5 \
    --batch_size 8 \
    --num_epochs 50 \
    --Class 2
```

---

## JSON 版 test_model 评估

```bash
python new_code/test_model.py \
    --pth_path ./modelsaved/Thyroid_PTC_vs_FTC_json/epoch_021_Thyroid_PTC_vs_FTC_json_V0.630_T0.509.pth \
    --test_json_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/test_labels.json \
    --valid_json_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/test_labels.json \
    --image_base_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped \
    --filename_key filename \
    --label_key FTCPTC \
    --ignore_labels -1 \
    --label_map 0:0,1:1 \
    --threshold_metric youden \
    --batch_size 8 \
    --num_workers 4 \
    --bootstrap_samples 1000 \
    --ci_level 0.95 \
    --seed 42 \
    --require_exists
```

---

## 说明

1. 单阶段生成更稳，建议先验证；两阶段生成更接近论文，但调试成本更高。
2. 缺失条件文件时会自动回退，因此更适合逐步接入真实条件图。
3. 切换到新版逻辑时，建议使用新的 `output_root`，避免混用旧缓存。

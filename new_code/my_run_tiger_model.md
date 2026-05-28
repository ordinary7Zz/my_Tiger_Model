# Tiger-Model：run_tiger_model.py 使用说明（旧版 JSON 流程）

## 适用场景

- 希望沿用当前稳定的 JSON 版分类数据构建流程
- 可选接入真实 mask / `condition_BG`
- 不需要 `condition_FG`，也不需要 `FG->BG` 两阶段生成

说明：

- 这是旧版增强流程，对应脚本：`new_code/run_tiger_model.py`
- 若你想使用新版单阶段 / 两阶段生成流程，请看 [my_run_tiger_model2.md](my_run_tiger_model2.md)

---

## 输入约定

1. JSON 顶层是 `list`，元素至少包含：
   - `filename`：图像相对路径（如 `2016/xxx/a.jpg`）
   - `FTCPTC`：标签（默认 `0/1`，`-1` 表示忽略）
2. 若提供真实 mask：
   - 通过 `--mask_dir` 指定 mask 根目录
   - mask 与原图保持相同相对路径、相同 stem
   - 扩展名可以不同（`.png` / `.jpg` / `.bmp` / ...）
3. 若提供 `condition_BG`：
   - 通过 `--condition_bg_dir` 指定目录
   - `condition_BG` 与原图保持相同相对路径、相同 stem
   - 一般建议先用 `new_code/make_condition_bg.py` 生成
4. 回退逻辑：
   - 缺 mask -> 回退到中心圆 mask
   - 缺 `condition_BG` -> 回退到原图 control

---

## 步骤 A：推荐先生成 condition_BG

### Ubuntu 小范围验证示例

```bash
python new_code/make_condition_bg.py \
    --json_path ./new_code/json/train_labels_small.json \
    --base_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped \
    --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped_predictionns \
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
    --json_path /mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/my_json/train_labels_filtered_by_csv.json \
    --base_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped \
    --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped_predictionns \
    --output_dir ./dataset_json/condition_BG \
    --filename_key filename \
    --size 512 \
    --mode hybrid \
    --dilate_px 5 \
    --require_exists
```

---

## 步骤 B：运行 JSON 版 run_tiger_model

输出目录默认在 `./dataset_json`：

- `Resnet_training_data/{train,valid,test}/{0,1}`
- `CSV/PTC_vs_FTC_{train,valid,test}.csv`

说明：

- `train/valid` 由 `train_labels.json` 切分得到
- 传入 `--test_json_path` 后，会额外生成独立 test 集
- 若提供 `--mask_dir / --condition_bg_dir`：
  - 优先使用真实 mask 和 `condition_BG`
  - 若某张图找不到对应文件，会自动回退到旧逻辑

### 示例 A：使用真实 mask + condition_BG 进行 Tiger 扩散增广（推荐）

```bash
python new_code/run_tiger_model.py \
    --json_path ./new_code/json/train_labels.json \
    --test_json_path ./new_code/json/test_labels.json \
    --base_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped \
    --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_masks \
    --condition_bg_dir ./dataset_json/condition_BG \
    --label_key FTCPTC \
    --ignore_labels -1 \
    --label_map 0:PTC:0,1:FTC:1 \
    --output_root ./dataset_json \
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

### 示例 B：兼容旧方法，不提供 mask_dir / condition_bg_dir

```bash
python new_code/run_tiger_model.py \
    --json_path ./new_code/json/train_labels.json \
    --test_json_path ./new_code/json/test_labels.json \
    --base_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/FangDai_Thyroid_Ultrasound_Images_cropped \
    --label_key FTCPTC \
    --ignore_labels -1 \
    --label_map 0:PTC:0,1:FTC:1 \
    --output_root ./dataset_json \
    --csv_prefix PTC_vs_FTC \
    --train_ratio 0.9 \
    --valid_ratio 0.1 \
    --seed 42 \
    --ptc_generate_per_image 0 \
    --ftc_generate_per_image 0 \
    --pretrain_model_path ./model/pretrain \
    --controlnet_bg_path ./model/fine-train-model/controlnet_bg \
    --require_exists
```

### 示例 C：使用类别平衡上限（按指定倍数下采样多数类）

无需 Tiger 模型：

```bash
python new_code/run_tiger_model.py \
    --json_path /mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/my_json/train_labels_filtered_by_csv.json \
    --test_json_path /mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/my_json/test_labels_filtered_by_csv.json \
    --base_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped \
    --label_key FTCPTC \
    --ignore_labels -1 \
    --label_map 0:PTC:0,1:FTC:1 \
    --output_root ./dataset_json \
    --csv_prefix PTC_vs_FTC \
    --train_ratio 0.9 \
    --valid_ratio 0.1 \
    --seed 42 \
    --cap_majority_ratio 1.0 \
    --require_exists
```

---

## 训练 ResNet 二分类

注意 `imagepath` 要对齐上面的 `output_root`（这里是 `./dataset_json`）。

```bash
python Resent/main_train.py \
    --modelname Thyroid_PTC_vs_FTC_json \
    --architecture resnet \
    --imagepath ./dataset_json \
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

### 新版：同时导出 txt + json

```bash
python new_code/test_model_export.py \
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

输出文件会保存在 `--test_json_path` 同目录下：

- `test_labels_metrics.txt`
- `test_labels_predictions.json`

---

## 说明

1. 使用真实 mask / `condition_BG` 时，建议使用新的 `output_root`，避免复用旧生成缓存。
2. 若使用 `--cap_majority_ratio`，则会忽略 `ptc_generate_per_image / ftc_generate_per_image`。
3. 这是兼容旧流程的版本；如果要更接近论文流程，请改用 `run_tiger_model2.py`。

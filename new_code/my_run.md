# Tiger-Model：PTC vs FTC 二分类流水线（JSON 版工作流备忘）

## 整体流程

按顺序执行：

1. 准备 JSON 标签文件（格式同 `train_labels.json`）
2. 准备真实结节 mask（与原图同相对路径 / 同文件 stem，扩展名可不同）
3. 生成 `condition_BG`（推荐，有真实 mask 时更接近论文流程）
4. 选择数据构建脚本：`run_tiger_model.py` 或 `run_tiger_model2.py`
5. ResNet 训练
6. 模型评估（JSON 测试集 + 可选 JSON 验证集阈值选择）

运行前请自行：

- 安装 Python 环境、PyTorch 与 requirements（见 `README` / `requirements_core.txt`）
- 将下面路径改成你本机路径（Windows 或 WSL 均可）
- 确保 JSON 中 `filename` 是相对于 `image_base_path/base_dir` 的相对路径

---

## 步骤 1：准备 JSON 标签文件

要求：JSON 顶层是 `list`，元素至少包含：

- `filename`：图像相对路径（如 `2016/xxx/a.jpg`）
- `FTCPTC`：标签（默认 `0/1`，`-1` 表示忽略）

示例文件：

- `./new_code/train_labels.json`
- `./new_code/valid_labels.json`
- `./new_code/test_labels.json`

---

## 步骤 2：准备真实结节 mask

约定：mask 不需要写入 JSON，`run_tiger_model.py`、`run_tiger_model2.py` 和 `make_condition_bg.py` 会自动推断：

- mask 根目录由 `--mask_dir` 提供
- mask 与原图保持相同相对路径、相同文件 stem
- 扩展名可以不同（`.png` / `.jpg` / `.bmp` / ... 都会自动查找）

例如：

- 原图：`/data/images/2016/A/case_001.jpg`
- mask：`/data/masks/2016/A/case_001.png`

---

## 步骤 3：生成 condition_BG（推荐）

说明：

- 输入：原图 + 真实结节 mask
- 输出：背景结构条件图 `condition_BG`
- 输出目录下保持与原图相同的相对路径，统一保存为 `png`
- 这一步只需要在你准备接入真实背景结构图时运行一次

建议先做小范围验证：

1. 先复制一份只包含少量样本的 JSON（例如 3~10 张）到 `./new_code/json/train_labels_small.json`
2. 先对这份小 JSON 运行 `make_condition_bg.py`
3. 打开输出图检查：
   - 结节区域是否被抹掉
   - 背景层次/边缘是否还在
   - 输出尺寸是否为 `512x512`

### 小范围验证示例

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

### 全量运行示例

```bash
python new_code/make_condition_bg.py \
    --json_path /mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/my_json/train_labels_filtered_by_csv.json \
    --base_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped \
    --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_predictions \
    --output_dir ./dataset_json/condition_BG \
    --filename_key filename \
    --size 512 \
    --mode hybrid \
    --dilate_px 5 \
    --require_exists
```

---

## 步骤 4：选择数据构建脚本

- 旧版流程（兼容原逻辑，支持真实 mask / `condition_BG`，可选 Tiger 增广）：见 [my_run_tiger_model.md](my_run_tiger_model.md)
- 新版流程（支持 `condition_FG`、单阶段生成、`FG->BG` 两阶段生成）：见 [my_run_tiger_model2.md](my_run_tiger_model2.md)

建议：

- 想保持当前稳定流程：看 [my_run_tiger_model.md](my_run_tiger_model.md)
- 想更接近论文流程：看 [my_run_tiger_model2.md](my_run_tiger_model2.md)

---

## 步骤 5：训练 ResNet 二分类

注意 `imagepath` 要对齐步骤 4 中你实际选择的 `output_root`。

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

## 步骤 6：JSON 版 test_model 评估

参数说明：

- `--test_json_path`：测试集 JSON（格式同 `train_labels.json`）
- `--valid_json_path`：验证集 JSON（可选，用于自动找阈值）
- `--image_base_path`：图像根目录，真实路径 = `image_base_path + filename`

```bash
python new_code/test_model.py \
    --pth_path ./modelsaved/Thyroid_PTC_vs_FTC_json_20260517_144828/epoch_013_Thyroid_PTC_vs_FTC_json_V0.623_T0.714.pth \
    --test_json_path /mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/my_json/test_labels_filtered_by_csv.json \
    --valid_json_path /mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/my_json/test_labels_filtered_by_csv.json \
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

## 常见排查

1. 报图像不存在：检查 `filename` 是否为相对路径，且能和 `base_dir/image_base_path` 拼接。
2. 报 mask 不存在：检查 `--mask_dir` 下是否有与原图相同相对路径、相同 stem 的 mask。
3. 报 `condition_BG` 不存在：先运行 `make_condition_bg.py`，或确认 `--condition_bg_dir` 路径是否正确。
4. 指标异常低：先确认 `test_json` 与训练数据没有泄露重叠。
5. 标签反了：调整 `--label_map`（如 `0:1,1:0`）。
6. JSON 有未标注：用 `--ignore_labels` 过滤（默认 `-1`）。
7. 切换到真实 mask / `condition_BG` / 新版生成逻辑时，建议使用新的 `output_root`，避免复用旧生成结果。

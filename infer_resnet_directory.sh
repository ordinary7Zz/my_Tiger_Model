#!/usr/bin/env bash
# 批量推理：infer_resnet_directory.py
# 用法：在项目根目录执行
#   bash infer_resnet_directory.sh
# 或直接：./infer_resnet_directory.sh（需 chmod +x）
# 若 Linux 报 bash\r：说明行尾是 Windows CRLF，在服务器执行：
#   sed -i 's/\r$//' infer_resnet_directory.sh

PTH_PATH="./modelsaved/weight.pth"
INPUT_DIR="./dataset/test"
OUTPUT_CSV="./output/predictions.csv"
NUM_CLASSES=2
BATCH_SIZE=4
DEVICE="cuda:0"

python infer_resnet_directory.py \
        --pth_path "$PTH_PATH" \
        --input_dir "$INPUT_DIR" \
        --output_csv "$OUTPUT_CSV" \
        --num_classes $NUM_CLASSES \
        --batch_size $BATCH_SIZE \
        --device $DEVICE

# 四分类示例（按需取消注释，并注释掉上面整块）
# PTH_PATH="./modelsaved/multiclass_xxx/best_model_weights.pth"
# INPUT_DIR="./dataset/your_test_images"
# OUTPUT_CSV="./output/predictions_4class.csv"
# NUM_CLASSES=4
# BATCH_SIZE=32
# DEVICE="cuda"
#
# python infer_resnet_directory.py \
#         --pth_path "$PTH_PATH" \
#         --input_dir "$INPUT_DIR" \
#         --output_csv "$OUTPUT_CSV" \
#         --num_classes $NUM_CLASSES \
#         --batch_size $BATCH_SIZE \
#         --device $DEVICE \
#         --recursive

python extract_radiomics_2d.py \
  --ptc_image_dir "/mnt/wangbd8/workspace/ThyroidAgent/Tiger-Model/dataset/new_data/subtype_1_processed_all" \
  --ptc_mask_dir "/mnt/wangbd8/workspace/ThyroidAgent/Tiger-Model/dataset/new_data/subtype_1_predictions" \
  --ftc_image_dir "/mnt/wangbd8/workspace/ThyroidAgent/Tiger-Model/dataset/new_data/subtype_2_processed_all" \
  --ftc_mask_dir "/mnt/wangbd8/workspace/ThyroidAgent/Tiger-Model/dataset/new_data/subtype_2_predictions" \
  --output_csv "output/radiomics_features.csv" \
  --params "./radiomics_2d.yaml"
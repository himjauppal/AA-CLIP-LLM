import os
import json

data_dir = "data"
output_dir = "dataset/metadata"
os.makedirs(output_dir, exist_ok=True)

for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    test_dir = os.path.join(class_path, "test")
    gt_dir = os.path.join(class_path, "ground_truth")

    if not os.path.exists(test_dir):
        continue

    output_path = os.path.join(output_dir, f"{class_name}.jsonl")
    with open(output_path, "w") as outfile:
        for defect_type in os.listdir(test_dir):
            defect_path = os.path.join(test_dir, defect_type)
            label = 0.0 if defect_type == "good" else 1.0

            for img_name in os.listdir(defect_path):
                if not img_name.endswith((".png", ".jpg", ".jpeg")):
                    continue

                record = {
                    "image_path": os.path.join("data", class_name, "test", defect_type, img_name),
                    "label": label,
                    "class_name": class_name
                }

                if label == 1.0:
                    mask_path = os.path.join("data", class_name, "ground_truth", defect_type, img_name)
                    if os.path.exists(os.path.join("/netscratch/uppal/AA-CLIP", mask_path)):
                        record["mask_path"] = mask_path

                outfile.write(json.dumps(record) + "\n")

    print(f"✅ Created: {output_path}")

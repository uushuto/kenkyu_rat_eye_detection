import json
import os
from collections import defaultdict
import shutil
from tqdm import tqdm

"""
detrがcoco形式のannotationファイルが必要なため,
labelme形式からcoco形式に変換するコード.

以下のようにデータが作成される.

├─── output_path
    ├─── annotations
    │       ├─── train.json
    │       └─── val.json
    ├─── trainImgages               
    └─── valImages

"""

# setting -------------------------------------------------------------------

# labelme形式jsonファイルが入っているフォルダ(画像.jptとannotationの.json)
folder_path = r"C:\Users\gijie\Rad_shuto\detr\ratkaiseki1"  

# 出力フォルダ
output_path = r"C:\Users\gijie\Rad_shuto\detr\coco_rad0-1600"

# annotationのlabel名のリスト
class_names = ["rat_eye"]

# 学習データを何割にするか,　ex: train_ratio = 0.8 学習データ,評価データを(8:2)に分ける
train_ratio = 0.8
# ----------------------------------------------------------------------------



def labelme_to_coco(json_and_img_folder, categories, train_ratio, output_path):

    json_files = [os.path.join(json_and_img_folder, f) for f in os.listdir(json_and_img_folder) if f.endswith('.json')]
    jsons_data = []
    for file in json_files:
        with open(file, 'r') as f:
            jsons_data.append(json.load(f))

    """Convert LabelMe data to COCO format."""
    data_num = len(jsons_data)
    train_data_number = int(data_num * train_ratio)

    coco_format_train = {
        "images": [],
        "annotations": [],
        "categories": categories
    }

    coco_format_val = {
        "images": [],
        "annotations": [],
        "categories": categories
    }

    annotation_train_id = 1
    annotation_val_id   = 1
    for img_id, item in tqdm(enumerate(jsons_data, start=1)):
        if img_id <= train_data_number:

            flg = 0
            for shape in item['shapes']:

                category_id = next((cat['id'] for cat in categories if cat['name'] == shape['label']), None)
                if category_id is None:
                    continue
                
                flg += 1

                points = shape['points']
                x_min, y_min = min([p[0] for p in points]), min([p[1] for p in points])
                x_max, y_max = max([p[0] for p in points]), max([p[1] for p in points])
                width, height = x_max - x_min, y_max - y_min

                coco_format_train["annotations"].append({
                    "id": annotation_train_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [x_min, y_min, width, height],
                    "area": width * height,
                    "iscrowd": 0
                })
                annotation_train_id += 1

            if flg >= 1:
                source_path = os.path.join(json_and_img_folder, item['imagePath'])
                destination_path = os.path.join(output_path, "trainImgages", item['imagePath'])
                shutil.copy(source_path, destination_path)

                coco_format_train["images"].append({
                    "file_name": item['imagePath'],
                    "height": item['imageHeight'],
                    "width": item['imageWidth'],
                    "id": img_id
                })
        else:

            flg = 0
            for shape in item['shapes']:
                category_id = next((cat['id'] for cat in categories if cat['name'] == shape['label']), None)
                if category_id is None:
                    continue

                flg += 1

                points = shape['points']
                x_min, y_min = min([p[0] for p in points]), min([p[1] for p in points])
                x_max, y_max = max([p[0] for p in points]), max([p[1] for p in points])
                width, height = x_max - x_min, y_max - y_min

                coco_format_val["annotations"].append({
                    "id": annotation_val_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [x_min, y_min, width, height],
                    "area": width * height,
                    "iscrowd": 0
                })
                annotation_val_id += 1

            if flg >= 1:
                source_path = os.path.join(json_and_img_folder, item['imagePath'])
                destination_path = os.path.join(output_path, "valImages", item['imagePath'])
                shutil.copy(source_path, destination_path)
                coco_format_val["images"].append({
                    "file_name": item['imagePath'],
                    "height": item['imageHeight'],
                    "width": item['imageWidth'],
                    "id": img_id
                })


    return coco_format_train, coco_format_val

out_anno_path = os.path.join(output_path, "annotations")
out_train_imgs_path = os.path.join(output_path, "trainImgages")
out_val_imgs_path   = os.path.join(output_path, "valImages")
if not os.path.exists(output_path):
    os.makedirs(output_path)
    os.makedirs(out_anno_path)
    os.makedirs(out_train_imgs_path)
    os.makedirs(out_val_imgs_path)
else:
    print("エラー： ", "すでに同じ名前のフォルダが存在します")
    exit()

categories = [{'id' : index, 'name': name} for index, name in enumerate(class_names)]

coco_train_data, coco_val_data = labelme_to_coco(folder_path, categories, train_ratio, output_path)


with open(os.path.join(out_anno_path, "train.json"), 'w') as f:
    json.dump(coco_train_data, f, indent=4)

with open(os.path.join(out_anno_path, "val.json"), 'w') as f:
    json.dump(coco_val_data, f, indent=4)




import math
import os
import sys
from typing import Iterable

import torch
import cv2
import numpy as np

import util.misc as utils

from datasets import build_dataset
import torch.nn as nn

from torch.utils.data import DataLoader
import datasets.transforms as T
from PIL import Image
import matplotlib.pyplot as plt


# 参考サイト
# https://www.ogis-ri.co.jp/otc/hiroba/technical/detr/part1.html
transforms = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# - setting --------------------------------------------------------------------------------
video_path = r".avi"  # 動画のパスを設定
model_path = r".pth"  # 学習済みモデルのパスを設定
output_video_path = r".avi"  # 出力する動画のパス
threshold = 0.9  # 信頼度の閾値
# ------------------------------------------------------------------------------------------

# モデルの設定
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
in_features = model.class_embed.in_features
model.class_embed = nn.Linear(in_features=in_features, out_features=1+1)
model.num_queries = 2
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model'])
model.eval()  # モデルを推論モードに設定

CLASSES = ["rat_eye"]
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# ボックスを変換する関数
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

# 動画処理の設定
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# フレームごとに処理
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # フレームをPILイメージに変換
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img, _ = transforms(pil_img, {})
    img = img.unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)

    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], pil_img.size)

    # フレームにバウンディングボックスを描画
    for p, (xmin, ymin, xmax, ymax), c in zip(probas[keep], bboxes_scaled.tolist(), COLORS):
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(int(c[0]*255), int(c[1]*255), int(c[2]*255)), thickness=2)
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        cv2.putText(frame, text, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (int(c[0]*255), int(c[1]*255), int(c[2]*255)), 2, cv2.LINE_AA)

    out.write(frame)
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

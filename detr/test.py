import math
import os
import sys
from typing import Iterable

import torch

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
    # T.Resize(800),
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# - setting --------------------------------------------------------------------------------
img_path =  r".jpg"
model_path = r".pth"
# 信頼度何以上を出力するか
threshold = 0.9
# ------------------------------------------------------------------------------------------


# modeの設定
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
in_features = model.class_embed.in_features
model.class_embed = nn.Linear(in_features=in_features, out_features=1+1)
model.num_queries = 2
checkpoint = torch.load(model_path)
print(checkpoint)
model.load_state_dict(checkpoint['model'])

# テストデータの設定
im = Image.open(img_path)
# dataset_val = im
# sampler_val = torch.utils.data.SequentialSampler(dataset_val)
# data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
#             drop_last=False, collate_fn=utils.collate_fn, num_workers=4)


img, _ = transforms(im, {})
img = img.unsqueeze(0)
outputs = model(img)

CLASSES = ["eye"]

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def box_cxcywh_to_xyxy(x):
    """
    (center_x, center_y, width, height)から(xmin, ymin, xmax, ymax)に座標変換
    """
    # unbind(1)でTensor次元を削除
    # (center_x, center_y, width, height)*N → (center_x*N, center_y*N, width*N, height*N)
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    # (center_x, center_y, width, height)*N の形に戻す
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    """
    バウンディングボックスのリスケール
    """
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    # バウンディングボックスの[0～1]から元画像の大きさにリスケール
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def plot_results(pil_img, prob, boxes):
    """
    画像とバウンディングボックスの表示
    """
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if prob is not None and boxes is not None:
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
            # バウンディングボックスの表示
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                         fill=False, color=c, linewidth=3))
            # 最大の予測値を持っているクラスのindexを取得
            cl = p.argmax()
            # クラス名と予測値の表示
            text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()



# no-objectを除いた91クラスでsoftmax
probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
# 91クラスの中で一番大きい予測値を取得*N個して、閾値を超えればTrue、それ以下だとFalse
keep = probas.max(-1).values > threshold
# バウンディングボックスの前処理
bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
# 画像とバウンディングボックスの表示
plot_results(im, probas[keep], bboxes_scaled)
# ラット眼球検出 (DETRベース)

このリポジトリでは、[DETR (DEtection TRansformer)](https://github.com/facebookresearch/detr) を使用したラット眼球検出モデルのトレーニングと評価を行うコードになります。 
本プロジェクトでは、COCO形式のデータセットを使用してモデルを学習および評価します。

---

## **リポジトリの目的**

- 本リポジトリは、研究で使用したコードの記録および説明を目的としています。
- 実行可能なコードを提供しますが、研究環境に特化しており、一般利用向けには設計されていません。

---

# コード詳細

- `main.py`：トレーニングするためのコードです。
- `test.py`：学習済みDETRモデルを使用してCOCO形式のデータセット上で評価を行うためのコードです。
- `test_video.py`：学習済みDETRモデルを使用して動画に対して評価を行うためのコードです。
- `labelme2coco.py`：Labelme形式のアノテーションデータをCOCO形式に変換するために使用されます。

---

## 環境構築

以下の手順で環境を構築してください。

### **1. 仮想環境の作成**

Anacondaを使用して仮想環境を作成します：

```bash
conda create --name rat_eye_detection python=3.9
conda activate rat_eye_detection

### **2. 必要なライブラリのインストール**
必要なPythonライブラリをインストールしてください。
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

## **実行方法**
1. モデルのトレーニング
以下のコマンドでモデルをトレーニングします：

python main.py --batch_size 8 --lr_drop 140 --output_dir ./logs --num_queries 1 --coco_path ./coco_annotations
引数の詳細
--batch_size: バッチサイズ（デフォルトは8）
--lr_drop: 学習率を変更するエポック数（例: 140）
--output_dir: モデルやログを保存するディレクトリ（例: ./logs）
--coco_path: COCO形式のデータセットへのパス（例: ./coco_annotations）

2. モデルの評価
学習済みモデルを使用して評価を行うには、以下のコマンドを実行します：

python test.py
---

## **参考文献**

1. **DETR (DEtection TRansformer)**  
   DETR公式リポジトリ: [https://github.com/facebookresearch/detr](https://github.com/facebookresearch/detr)

2. **COCOデータセット**  
   COCOデータセット公式サイト: [https://cocodataset.org/#home](https://cocodataset.org/#home)

3. **PyTorch公式サイト**  
   PyTorchの公式ドキュメント: [https://pytorch.org/](https://pytorch.org/)

4. **Python公式サイト**  
   Pythonダウンロードページ: [https://www.python.org/downloads/](https://www.python.org/downloads/)

5. **Labelme**  
   Labelme GitHubリポジトリ: [https://github.com/wkentaro/labelme](https://github.com/wkentaro/labelme)

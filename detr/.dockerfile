# ベースイメージとしてCUDA 11.8とPython 3.9を指定
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# 作業ディレクトリを設定
WORKDIR /app

# 必要なシステムパッケージをインストール
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-distutils \
    python3-pip \
    wget \
    git && \
    apt-get clean

# pipを最新バージョンにアップグレード
RUN pip install --upgrade pip

# 必要なライブラリをインストール
COPY requirements.txt .
RUN pip install -r requirements.txt

# プロジェクトコードをコピー
COPY . .

# コンテナ実行時のデフォルトコマンドを設定
CMD ["python", "main.py"]

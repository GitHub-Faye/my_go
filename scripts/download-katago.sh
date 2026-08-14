#!/usr/bin/env bash
# 下载 KataGo 静态分析模型（uint8 量化，~75MB）到 my_go/models/。
# 模型 URL 与原版 Kaya 引擎一致（kaya-go/kaya HF 仓库，tag 0.2.2）。
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_NAME="kata1-b28c512nbt-s12043015936-d5616446734"
QUANT="uint8"  # fp32 / fp16 / uint8 可选
URL="https://huggingface.co/kaya-go/kaya/resolve/0.2.2/${MODEL_NAME}/${MODEL_NAME}.${QUANT}.onnx"
OUT="models/kata1-${QUANT}.onnx"

mkdir -p models
if [ -f "$OUT" ]; then
  echo "已存在：$OUT（跳过）"
  exit 0
fi

echo "下载 $QUANT 模型 ${URL} ..."
curl -L --fail --progress-bar -o "$OUT" "$URL"
echo "完成：$OUT"

#!/usr/bin/env bash
# Sync CoreML model bundle and tokenizer/config files into the iOS app resources.

set -euo pipefail

DEST_DEFAULT="TestCoreML/TestCoreML/NLPResources"
MODEL_PACKAGE=${MODEL_PACKAGE:-coreml_model.mlpackage}
MERGED_DIR=${MERGED_DIR:-merged_model}
TOKENIZER_DIR=${TOKENIZER_DIR:-lora_adapter}

DEST_DIR=${1:-$DEST_DEFAULT}

ARTIFACTS=(
  "$MODEL_PACKAGE|coreml_model.mlpackage"
  "$MERGED_DIR/config.json|config.json"
  "$TOKENIZER_DIR/tokenizer.json|tokenizer.json"
  "$TOKENIZER_DIR/tokenizer_config.json|tokenizer_config.json"
  "$TOKENIZER_DIR/special_tokens_map.json|special_tokens_map.json"
  "$TOKENIZER_DIR/vocab.txt|vocab.txt"
)

mkdir -p "$DEST_DIR"

for entry in "${ARTIFACTS[@]}"; do
  SRC_PATH=${entry%%|*}
  DEST_NAME=${entry##*|}
  if [[ ! -e "$SRC_PATH" ]]; then
    echo "[copy_model] 缺少文件: $SRC_PATH" >&2
    exit 1
  fi

  DEST_PATH="$DEST_DIR/$DEST_NAME"
  rm -rf "$DEST_PATH"
  cp -R "$SRC_PATH" "$DEST_PATH"
  echo "[copy_model] 拷贝 $SRC_PATH -> $DEST_PATH"
done

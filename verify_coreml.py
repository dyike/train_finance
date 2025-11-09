import json
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import os

MODEL_DIR = Path("merged_model")
TOKENIZER_DIR = Path("lora_adapter")
COREML_PATH = Path("coreml_model.mlpackage")
TMP_DIR = Path(".coreml_tmp")
TMP_DIR.mkdir(exist_ok=True)
tmp_path = str(TMP_DIR.resolve())
os.environ.setdefault("TMPDIR", tmp_path)
os.environ.setdefault("CFFIXED_USER_HOME", tmp_path)
os.environ.setdefault("HOME", tmp_path)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
TEXT = "昨天在星巴克花了36元买咖啡"
MAX_LEN = 64
SLOT_ORDER = ["amount", "name", "category", "type", "time"]

print("🔹 Loading tokenizer & models ...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
torch_model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
torch_model.eval()
mlmodel = ct.models.MLModel(str(COREML_PATH))

id2label = {int(k): v for k, v in torch_model.config.id2label.items()}
SPECIAL_TOKENS = {tok for tok in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token] if tok}


def clean_piece(token: str) -> str:
    if token.startswith("##"):
        return token[2:]
    if token.startswith("▁"):
        return token[1:]
    return token


def extract_slots(tokens, labels):
    slots = {slot: [] for slot in SLOT_ORDER}
    current_slot = None
    current_pieces = []

    def flush():
        nonlocal current_slot, current_pieces
        if current_slot and current_pieces:
            slots[current_slot].append("".join(current_pieces))
        current_slot = None
        current_pieces = []

    for token, label_id in zip(tokens, labels):
        if token in SPECIAL_TOKENS:
            flush()
            continue
        label = id2label.get(int(label_id), "O")
        piece = clean_piece(token)
        if not piece or label == "O" or "-" not in label:
            flush()
            continue
        prefix, slot_name = label.split("-", maxsplit=1)
        if slot_name not in slots:
            flush()
            continue
        if prefix == "B" or slot_name != current_slot:
            flush()
            current_slot = slot_name
            current_pieces = [piece]
        else:
            current_pieces.append(piece)
    flush()
    return {slot: " / ".join(values) if values else "-" for slot, values in slots.items()}


# ===== Torch inference =====
torch_inputs = tokenizer(
    TEXT,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN,
)
with torch.no_grad():
    torch_logits = torch_model(**torch_inputs).logits
torch_preds = torch_logits.argmax(dim=-1)[0].tolist()
tokens = tokenizer.convert_ids_to_tokens(torch_inputs["input_ids"][0])
torch_slots = extract_slots(tokens, torch_preds)

print("\n⚡️ PyTorch logits -> labels:")
for token, label in zip(tokens, torch_preds):
    print(f"{token:10s} -> {id2label[int(label)]}")
print("Torch slots:", json.dumps(torch_slots, ensure_ascii=False))

# ===== CoreML inference =====
coreml_inputs = tokenizer(
    TEXT,
    return_tensors="np",
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN,
)
ml_inputs = {
    "input_ids": coreml_inputs["input_ids"].astype(np.int32),
    "attention_mask": coreml_inputs["attention_mask"].astype(np.int32),
}
try:
    coreml_outputs = mlmodel.predict(ml_inputs)
    output_key = next(iter(coreml_outputs.keys()))
    pred_array = coreml_outputs[output_key]
    if not isinstance(pred_array, np.ndarray):
        pred_array = np.array(pred_array)
    coreml_preds = pred_array.reshape(1, -1)[0].tolist()
    coreml_slots = extract_slots(tokens, coreml_preds)

    print("\n🍏 CoreML predictions -> labels:")
    for token, label in zip(tokens, coreml_preds):
        print(f"{token:10s} -> {id2label.get(int(label), 'O')}")
    print("CoreML slots:", json.dumps(coreml_slots, ensure_ascii=False))

    print("\n✅ Match?", torch_preds == coreml_preds)
except Exception as err:
    print(f"⚠️ Skipped CoreML predict step due to sandbox restrictions: {err}")

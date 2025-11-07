import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# ===== 1. 加载模型 =====
MODEL_DIR = "./merged_model"  # merge_lora.py 输出的路径
TOKENIZER_DIR = "./lora_adapter"  # Tokenizer / LoRA 保存目录

print("🔹 Loading model ...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
model.eval()

id2label = {int(k): v for k, v in model.config.id2label.items()}
SLOT_ORDER = ["amount", "name", "category", "type", "time"]
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
        if not piece:
            continue

        if label == "O" or "-" not in label:
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


# ===== 2. 测试输入 =====
# text = "昨天在星巴克花了36元买咖啡"
text = "在星巴克用了18块钱"
inputs = tokenizer(text, return_tensors="pt")

# ===== 3. 推理 =====
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)[0].tolist()

# ===== 4. 输出结果 =====
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
slots = extract_slots(tokens, predictions)

print("\n🔍 token 标签:")
for token, label_id in zip(tokens, predictions):
    label = id2label.get(int(label_id), "O")
    print(f"{token:10s} -> {label}")

print("\n📦 槽位解析:")
for slot in SLOT_ORDER:
    print(f"{slot:8s}: {slots[slot]}")

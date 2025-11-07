from peft import PeftModel
from transformers import AutoModelForTokenClassification


SLOT_NAMES = ["amount", "name", "category", "type", "time"]
LABEL_LIST = ["O"]
for slot in SLOT_NAMES:
    LABEL_LIST.append(f"B-{slot}")
    LABEL_LIST.append(f"I-{slot}")

label2id = {label: idx for idx, label in enumerate(LABEL_LIST)}
id2label = {idx: label for label, idx in label2id.items()}


base = AutoModelForTokenClassification.from_pretrained(
    "hfl/rbt3",
    num_labels=len(LABEL_LIST),
    id2label=id2label,
    label2id=label2id,
)
lora = PeftModel.from_pretrained(base, "./lora_adapter")
merged = lora.merge_and_unload()
merged.save_pretrained("./merged_model")

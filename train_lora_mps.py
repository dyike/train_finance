import inspect
import re
import numpy as np

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model


# ===== 1. 定义标签映射 =====
SLOT_NAMES = ["amount", "name", "category", "type", "time"]
LABEL_LIST = ["O"]
for slot in SLOT_NAMES:
    LABEL_LIST.append(f"B-{slot}")
    LABEL_LIST.append(f"I-{slot}")

label2id = {label: idx for idx, label in enumerate(LABEL_LIST)}
id2label = {idx: label for label, idx in label2id.items()}


# ===== 2. 加载模型与数据 =====
model_name = "hfl/rbt3"  # 中文 TinyBERT
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if not getattr(tokenizer, "is_fast", False):
    raise ValueError("当前脚本需要 fast tokenizer 以使用 offset mapping")
raw_dataset = load_dataset("csv", data_files="data/train_30000.csv")
dataset = raw_dataset["train"].train_test_split(test_size=0.1, seed=42)


def _find_all_spans(text: str, value: str):
    """返回 text 中 value 的所有起止下标（闭开区间）。"""
    if not value or value.lower() == "nan":
        return []
    spans = []
    for match in re.finditer(re.escape(str(value)), text):
        start, end = match.span()
        spans.append((start, end))
    return spans


def tokenize_and_align(batch):
    encodings = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=96,
        return_offsets_mapping=True,
    )

    labels = []
    for idx, offsets in enumerate(encodings["offset_mapping"]):
        text = batch["text"][idx]
        slot_values = {slot: batch[slot][idx] for slot in SLOT_NAMES}

        # 预置 O/-100 标签
        token_labels = []
        for start, end in offsets:
            if start == end:
                token_labels.append(-100)  # special/padding token
            else:
                token_labels.append(label2id["O"])

        # 为每个槽位打标签
        for slot_name, value in slot_values.items():
            spans = _find_all_spans(text, str(value).strip())
            for span_start, span_end in spans:
                began = False
                for token_idx, (token_start, token_end) in enumerate(offsets):
                    if token_start == token_end:
                        continue
                    if token_end <= span_start:
                        continue
                    if token_start >= span_end:
                        break
                    label_prefix = "B" if not began else "I"
                    token_labels[token_idx] = label2id[f"{label_prefix}-{slot_name}"]
                    began = True

        labels.append(token_labels)

    encodings["labels"] = labels
    encodings.pop("offset_mapping")
    return encodings


tokenized = dataset.map(
    tokenize_and_align,
    batched=True,
    remove_columns=raw_dataset["train"].column_names,
)


# ===== 3. 加载预训练模型 =====
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(LABEL_LIST),
    id2label=id2label,
    label2id=label2id,
)


# ===== 4. 应用 LoRA =====
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.05,
    bias="none",
    modules_to_save=["classifier"],
)
model = get_peft_model(model, lora_config)


# ===== 5. 训练参数 =====
training_args_kwargs = dict(
    output_dir="./output_lora",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    learning_rate=3e-4,
    logging_steps=20,
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=False,
    report_to="none",
)

training_args_signature = inspect.signature(TrainingArguments.__init__).parameters
if "eval_strategy" in training_args_signature:
    training_args_kwargs["eval_strategy"] = "epoch"
else:
    training_args_kwargs["evaluation_strategy"] = "epoch"

args = TrainingArguments(**training_args_kwargs)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    mask = labels != -100
    accuracy = (predictions == labels)[mask].mean() if mask.any() else 0.0
    return {"token_accuracy": round(float(accuracy), 4)}


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# ===== 6. 开始训练 =====
trainer.train()


# ===== 7. 保存 LoRA Adapter =====
model.save_pretrained("./lora_adapter")
tokenizer.save_pretrained("./lora_adapter")

# export_coreml.py  —— TS-friendly 包装 + trace + CoreML 转换（CPU，更稳）
import torch
import coremltools as ct
from pathlib import Path
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_DIR = Path("./merged_model")
TOKENIZER_DIR = Path("./lora_adapter")
OUT_PATH = Path("coreml_model.mlpackage")

print("🔹 Loading HF model:", MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

# ---- 关键：包装成 TS 友好的 Module，避免 dict / **kwargs ----
class TSWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor):
        # return_dict=False 会返回 tuple，位置 0 是 logits
        logits = self.m(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0]
        return logits.float()  # 明确成 float32，避免 dtype 兼容问题

model.eval()
wrapper = TSWrapper(model)

# ---- 用 CPU trace 更稳（避免 MPS trace 卡住）----
device = torch.device("cpu")
wrapper.to(device)

# 构造稳定示例输入
text = "昨天星巴克花了36元买咖啡"
example = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
input_ids = example["input_ids"].to(device)
attention_mask = example["attention_mask"].to(device)

print("🔹 Sanity forward:", wrapper(input_ids, attention_mask).shape)

print("🔹 Tracing TorchScript ...")
with torch.no_grad():
    traced = torch.jit.trace(wrapper, (input_ids, attention_mask), strict=False)
print("✅ trace ok")

# ---- Core ML 转换 ----
print("🔹 Converting to CoreML ...")
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="input_ids", shape=input_ids.shape, dtype=int),
        ct.TensorType(name="attention_mask", shape=attention_mask.shape, dtype=int),
    ],
    convert_to="mlprogram",
    compute_units=ct.ComputeUnit.CPU_AND_NE,  # 先用 CPU+NE，兼容性更好；需要可改 .ALL
    minimum_deployment_target=ct.target.iOS17,
    compute_precision=ct.precision.FLOAT16,   # 若遇到不兼容可改成 FLOAT32
)
mlmodel.save(OUT_PATH)
print(f"✅ CoreML model saved -> {OUT_PATH.resolve()}")

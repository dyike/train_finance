import os
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

TMP_DIR = Path(".coreml_tmp")
TMP_DIR.mkdir(exist_ok=True)
tmp_path = str(TMP_DIR.resolve())
os.environ.setdefault("TMPDIR", tmp_path)
os.environ.setdefault("CFFIXED_USER_HOME", tmp_path)
os.environ.setdefault("HOME", tmp_path)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_DIR = Path("./merged_model")
TOKENIZER_DIR = Path("./lora_adapter")
OUT_PATH = Path(os.environ.get("FINANCE_COREML_OUT", "coreml_model.mlpackage"))
SKIP_TEST_SAVE = os.environ.get("FINANCE_SKIP_TEST_SAVE") == "1"

print("🔹 Loading HF model:", MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

class TSWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    
    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor):
        logits = self.m(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0]
        return logits.float()

model.eval()
wrapper = TSWrapper(model)

device = torch.device("cpu")
wrapper.to(device)

# 构造测试样本
text = "昨天星巴克花了36元买咖啡"
example = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
input_ids = example["input_ids"].to(device)
attention_mask = example["attention_mask"].to(device)

# 保存测试数据用于 iOS 端对比
if SKIP_TEST_SAVE:
    print("🔹 Skipping test sample save (FINANCE_SKIP_TEST_SAVE=1)")
else:
    print("🔹 Saving test data ...")
    torch.save({
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'text': text
    }, 'test_sample.pt')

print("🔹 PyTorch prediction:", wrapper(input_ids, attention_mask))

print("🔹 Tracing TorchScript ...")
with torch.no_grad():
    traced = torch.jit.trace(wrapper, (input_ids, attention_mask), strict=False)

# 验证 traced 模型
traced_output = traced(input_ids, attention_mask)
wrapper_output = wrapper(input_ids, attention_mask)
print("🔹 Trace validation:", torch.allclose(traced_output, wrapper_output))

print("🔹 Converting to CoreML ...")
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="input_ids", shape=input_ids.shape, dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=attention_mask.shape, dtype=np.int32),
    ],
    outputs=[ct.TensorType(name="logits", dtype=np.float32)],
    convert_to="mlprogram",
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.iOS17,
    compute_precision=ct.precision.FLOAT32,
)

# 验证 CoreML 模型
print("🔹 Validating CoreML model ...")
coreml_input = {
    'input_ids': input_ids.numpy().astype(np.int32),
    'attention_mask': attention_mask.numpy().astype(np.int32)
}
try:
    coreml_output = mlmodel.predict(coreml_input)
    print("🔹 CoreML prediction:", coreml_output)
except Exception as err:
    print(f"⚠️ Skipped local CoreML predict check: {err}")

mlmodel.save(OUT_PATH)
print(f"✅ CoreML model saved -> {OUT_PATH.resolve()}")

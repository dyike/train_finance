"""Generate a large synthetic NER dataset where every slot value
appears verbatim inside the sentence.

Steps:
1. Load the tiny seed samples in data/train.csv.
2. Pick a template that mentions amount/name/category/type/time in plain text.
3. Repeat until we have the desired number of records and write them out.
"""

import pandas as pd
import random
from tqdm import tqdm

# ===== 输入输出配置 =====
SRC_FILE = "data/train.csv"        # 你的原始csv
OUT_FILE = "data/train_30000.csv"     # 输出路径
SAMPLE_COUNT = 30_000             # 目标数量

# ===== 读取原始数据 =====
df = pd.read_csv(SRC_FILE)

# ===== 模板池（确保所有槽位都在文本里出现） =====
EXPENSE_TEMPLATES = [
    "{time}在{name}花了{amount}元，用于{category}{type}",
    "{time}去{name}{type}{amount}元，全部花在{category}",
    "{time}向{name}支付{amount}元，归为{category}{type}",
    "{time}在{name}消费{amount}元，记在{category}{type}",
    "{time}给{name}{type}{amount}元，花在{category}",
    "{time}和{name}发生{type}{amount}元，属于{category}",
]

INCOME_TEMPLATES = [
    "{time}从{name}获得{amount}元{type}，归为{category}",
    "{time}{type}{amount}元来自{name}，标记为{category}",
    "{time}收到{name}打来的{amount}元{type}，分类{category}",
    "{time}由{name}汇入{amount}元，为{category}{type}",
    "{time}{type}{amount}元，来源{name}，记作{category}",
    "{time}和{name}之间的{type}{amount}元算作{category}",
]

DEFAULT_TEMPLATES = [
    "{time}在{name}{type}{amount}元，用于{category}",
]


def pick_template(row_type: str) -> str:
    if row_type == "支出":
        return random.choice(EXPENSE_TEMPLATES)
    if row_type == "收入":
        return random.choice(INCOME_TEMPLATES)
    return random.choice(DEFAULT_TEMPLATES)


# ===== 核心生成函数 =====
def make_sample(row):
    template = pick_template(row["type"])
    text = template.format(
        name=row["name"],
        category=row["category"],
        amount=row["amount"],
        time=row["time"],
        type=row["type"],
    )
    return {
        "text": text,
        "amount": row["amount"],
        "name": row["name"],
        "category": row["category"],
        "type": row["type"],
        "time": row["time"],
    }

# ===== 扩充生成 =====
rows = df.to_dict("records")
dataset = [make_sample(random.choice(rows)) for _ in tqdm(range(SAMPLE_COUNT), desc="🔧 生成样本")]

# ===== 保存结果 =====
out_df = pd.DataFrame(dataset)
out_df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
print(f"✅ 已生成 {SAMPLE_COUNT} 条训练数据 -> {OUT_FILE}")

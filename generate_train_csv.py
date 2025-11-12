"""Generate a large synthetic NER dataset where every slot value
appears verbatim inside the sentence.

Steps:
1. Load the tiny seed samples in data/train.csv.
2. Pick a template that mentions amount/name/category/type/time in plain text.
3. Repeat until we have the desired number of records and write them out.
"""

import random

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm missing
    def tqdm(iterator, **_):
        return iterator

try:
    import pandas as pd
except ImportError:  # pragma: no cover - 提示用户安装
    pd = None


TIME_DAY_DESCRIPTORS = [
    "今天",
    "昨天",
    "前天",
    "本周一",
    "本周三",
    "本周五",
    "上周二",
    "上周四",
    "上周末",
    "本月1号",
    "本月15号",
    "本月月底",
    "上个月初",
    "上个月底",
    "去年12月25日",
    "今年3月10日",
]

TIME_PERIOD_WORDS = ["清晨", "早上", "上午", "中午", "下午", "傍晚", "晚上", "深夜", "凌晨"]
TIME_MINUTE_CHOICES = list(range(0, 60, 5))
TIME_PATTERNS = [
    "{day}{period}{hour}点{minute:02d}分",
    "{day}{hour}点{minute:02d}",
    "{day}{period}{hour}点整",
    "{day}{period}{hour}:{minute:02d}",
    "{day}{period}{hour}点",
    "{day}{hour:02d}:{minute:02d}",
    "{day}{period}{hour}点半",
]

AMOUNT_VARIATION_RANGE = (0.85, 1.15)
AMOUNT_DECIMAL_WEIGHTS = [2, 3, 3]  # prefer decimals but still keep integers
SPOKEN_AMOUNT_RATIO = 0.55


# ===== 输入输出配置 =====
SRC_FILE = "data/train_src.csv"        # 你的原始csv
OUT_FILE = "data/train.csv"     # 输出路径
SAMPLE_COUNT = 50_000             # 目标数量

# ===== 模板池（确保所有槽位都在文本里出现） =====
EXPENSE_TEMPLATES = [
    "{time}在{name}花了{amount}，用于{category}{type}",
    "{time}去{name}{type}{amount}，全部花在{category}",
    "{time}向{name}支付{amount}，归为{category}{type}",
    "{time}在{name}消费{amount}，记在{category}{type}",
    "{time}给{name}{type}{amount}，花在{category}",
    "{time}和{name}发生{type}{amount}，属于{category}",
]

INCOME_TEMPLATES = [
    "{time}从{name}获得{amount}{type}，归为{category}",
    "{time}{type}{amount}来自{name}，标记为{category}",
    "{time}收到{name}打来的{amount}{type}，分类{category}",
    "{time}由{name}汇入{amount}，为{category}{type}",
    "{time}{type}{amount}，来源{name}，记作{category}",
    "{time}和{name}之间的{type}{amount}算作{category}",
]

DEFAULT_TEMPLATES = [
    "{time}在{name}{type}{amount}，用于{category}",
]


def normalize_text(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    if pd is not None:
        try:
            if pd.isna(value):  # handles pandas/numpy scalar NaN
                return ""
        except (TypeError, ValueError):
            pass
    return str(value).strip()


def dedup_preserve(values):
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def value_to_cents(value: float) -> int:
    return max(1, int(round(value * 100)))


def choose_numeric_amount(value: float) -> str:
    decimals_choice = random.choices([0, 1, 2], weights=AMOUNT_DECIMAL_WEIGHTS, k=1)[0]
    if decimals_choice == 0:
        int_value = int(round(value))
        suffix = "元整" if random.random() < 0.3 else "元"
        return f"{int_value}{suffix}"
    formatted = f"{value:.{decimals_choice}f}"
    return f"{formatted}元"


def build_spoken_amount_options(value: float):
    cents = value_to_cents(value)
    yuan = cents // 100
    mao = (cents % 100) // 10
    fen = cents % 10

    options = []
    if yuan:
        options.append(f"{yuan}块")
        if mao:
            base = f"{yuan}块{mao}毛"
            options.append(base)
            if fen:
                options.append(f"{base}{fen}")
                options.append(f"{base}{fen}分")
            else:
                options.append(f"{base}整")
        if not mao and fen:
            options.append(f"{yuan}块{fen}分")
            options.append(f"{yuan}块{fen}")
        if mao == 0 and fen == 0:
            options.append(f"{yuan}块整")

        if mao or fen:
            formal = f"{yuan}元"
            if mao:
                formal += f"{mao}角"
            if fen:
                formal += f"{fen}分"
            options.append(formal)
    else:
        if mao:
            base = f"{mao}毛"
            options.append(base)
            if fen:
                options.append(f"{base}{fen}")
                options.append(f"{base}{fen}分")
            else:
                options.append(f"{base}整")
        if fen:
            options.append(f"{fen}分")

    return dedup_preserve([opt for opt in options if opt])


def choose_spoken_amount(value: float) -> str | None:
    options = build_spoken_amount_options(value)
    if not options:
        return None
    return random.choice(options)


def split_seed_time(seed: str) -> tuple[str, str]:
    if not seed:
        return "", ""
    for period in TIME_PERIOD_WORDS:
        if period in seed:
            day_part = seed.replace(period, "").strip()
            return day_part, period
    return seed, ""


def random_detailed_time(seed_time: str) -> str:
    seed_time = normalize_text(seed_time)
    if seed_time and random.random() < 0.3:
        return seed_time

    seed_day, seed_period = split_seed_time(seed_time)
    day_candidates = TIME_DAY_DESCRIPTORS.copy()
    period_candidates = TIME_PERIOD_WORDS.copy()

    if seed_day:
        day_candidates.append(seed_day)
    if seed_period:
        period_candidates.append(seed_period)

    day = random.choice(day_candidates)
    period = random.choice(period_candidates)
    hour = random.randint(6, 22)
    minute = random.choice(TIME_MINUTE_CHOICES)
    pattern = random.choice(TIME_PATTERNS)
    return pattern.format(day=day, period=period, hour=hour, minute=minute)


def randomize_amount(raw_amount) -> str:
    raw_str = normalize_text(raw_amount).replace("元", "").replace(",", "")
    try:
        base_value = float(raw_str)
    except ValueError:
        base_value = random.uniform(10, 500)

    multiplier = random.uniform(*AMOUNT_VARIATION_RANGE)
    new_value = max(0.01, base_value * multiplier)
    if random.random() < SPOKEN_AMOUNT_RATIO:
        spoken_amount = choose_spoken_amount(new_value)
        if spoken_amount:
            return spoken_amount

    return choose_numeric_amount(new_value)


def pick_template(row_type: str) -> str:
    if row_type == "支出":
        return random.choice(EXPENSE_TEMPLATES)
    if row_type == "收入":
        return random.choice(INCOME_TEMPLATES)
    return random.choice(DEFAULT_TEMPLATES)


# ===== 核心生成函数 =====
def make_sample(row):
    amount = randomize_amount(row["amount"])
    time_value = random_detailed_time(row["time"])
    template = pick_template(row["type"])
    text = template.format(
        name=row["name"],
        category=row["category"],
        amount=amount,
        time=time_value,
        type=row["type"],
    )
    return {
        "text": text,
        "amount": amount,
        "name": row["name"],
        "category": row["category"],
        "type": row["type"],
        "time": time_value,
    }


def build_dataset(rows, sample_count):
    return [make_sample(random.choice(rows)) for _ in tqdm(range(sample_count), desc="🔧 生成样本")]


def main():
    if pd is None:
        raise ImportError("需要先安装 pandas 才能生成数据 (pip install pandas)")
    df = pd.read_csv(SRC_FILE)
    rows = df.to_dict("records")
    dataset = build_dataset(rows, SAMPLE_COUNT)

    out_df = pd.DataFrame(dataset)
    out_df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"✅ 已生成 {SAMPLE_COUNT} 条训练数据 -> {OUT_FILE}")


if __name__ == "__main__":
    main()

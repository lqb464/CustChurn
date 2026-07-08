"""Tạo năm notebook phân tích LendScope theo một cấu trúc có thể tái lập.

Các notebook được sinh bằng nbformat để phần source dễ review. Sau khi sinh,
notebook phải được thực thi bằng kernel sạch và lưu output trước khi bàn giao.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def write_notebook(name: str, cells: list) -> None:
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.14"},
        },
    )
    nbf.write(notebook, NB_DIR / name)


COMMON_SETUP = r'''
from pathlib import Path
import os, sys, json, warnings

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
os.environ["MPLCONFIGDIR"] = str(ROOT / ".matplotlib")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 80)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
sns.set_theme(style="whitegrid", context="notebook")

TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"
TARGET = "Loan Sanction Amount (USD)"
REQUEST = "Loan Amount Request (USD)"
RANDOM_STATE = 42

df_raw = pd.read_csv(TRAIN_PATH)
df_external = pd.read_csv(TEST_PATH)
print(f"Train: {df_raw.shape[0]:,} dòng x {df_raw.shape[1]} cột")
print(f"Test ngoài: {df_external.shape[0]:,} dòng x {df_external.shape[1]} cột")
'''


FEATURE_FUNCTIONS = r'''
SENTINEL = -999
DROP_COLUMNS = ["Customer ID", "Name", "Property ID", "Gender"]
BASE_FEATURES = [
    "Age", "Income (USD)", "Income Stability", "Profession",
    "Type of Employment", "Location", REQUEST,
    "Current Loan Expenses (USD)", "Expense Type 1", "Expense Type 2",
    "Dependents", "Credit Score", "No. of Defaults",
    "Has Active Credit Card", "Property Age", "Property Type",
    "Property Location", "Co-Applicant", "Property Price",
]
ENGINEERED_FEATURES = [
    "RequestToIncome", "RequestToProperty", "ExpenseToIncome",
    "IncomePerDependent", "PropertyEquityProxy", "CreditIncomeInteraction",
    "CreditRequestInteraction", "AgeIncomeInteraction",
]

def normalize_missing(df):
    out = df.copy()
    out = out.replace(SENTINEL, np.nan)
    return out

def add_business_features(df):
    out = normalize_missing(df)
    eps = 1e-6
    income = out["Income (USD)"].clip(lower=eps)
    request = out[REQUEST].clip(lower=0)
    property_price = out["Property Price"].clip(lower=eps)
    expenses = out["Current Loan Expenses (USD)"].clip(lower=0)
    dependents = out["Dependents"].clip(lower=0).fillna(0)
    credit = out["Credit Score"]

    out["RequestToIncome"] = request / income
    out["RequestToProperty"] = request / property_price
    out["ExpenseToIncome"] = expenses / income
    out["IncomePerDependent"] = income / (dependents + 1)
    out["PropertyEquityProxy"] = property_price - request
    out["CreditIncomeInteraction"] = credit * np.log1p(income)
    out["CreditRequestInteraction"] = credit / (request + 1) * 1000
    out["AgeIncomeInteraction"] = out["Age"] * np.log1p(income)
    return out

def modeling_frame(df):
    out = add_business_features(df)
    columns = [c for c in BASE_FEATURES + ENGINEERED_FEATURES if c in out.columns]
    return out[columns].copy()

def valid_regression_rows(df):
    target = pd.to_numeric(df[TARGET], errors="coerce")
    return target.notna() & (target > 0)
'''


SPLIT_FUNCTIONS = r'''
from sklearn.model_selection import train_test_split

def quantile_labels(y, q=10):
    return pd.qcut(y.rank(method="first"), q=q, labels=False, duplicates="drop")

def make_splits(df, random_state=RANDOM_STATE):
    clean = normalize_missing(df.loc[valid_regression_rows(df)].copy())
    train_val, test = train_test_split(
        clean,
        test_size=0.15,
        random_state=random_state,
        stratify=quantile_labels(clean[TARGET]),
    )
    train, val = train_test_split(
        train_val,
        test_size=0.1764705882,
        random_state=random_state,
        stratify=quantile_labels(train_val[TARGET]),
    )
    return train.copy(), val.copy(), test.copy()
'''


MODEL_FUNCTIONS = r'''
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    mean_squared_log_error,
)

def build_preprocessor(X, scale_numeric=False):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    numeric_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        numeric_steps.append(("scaler", RobustScaler()))
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor

def regression_metrics(y_true, y_pred, requested=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_error = np.abs(y_true - y_pred)
    denom = np.maximum(np.abs(y_true), 1.0)
    result = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "rmsle": np.sqrt(mean_squared_log_error(np.maximum(y_true, 0), np.maximum(y_pred, 0))),
        "wape": abs_error.sum() / np.abs(y_true).sum(),
        "within_10pct": np.mean(abs_error / denom <= 0.10),
        "within_20pct": np.mean(abs_error / denom <= 0.20),
        "mean_signed_error": np.mean(y_pred - y_true),
        "over_rate": np.mean(y_pred > y_true),
    }
    if requested is not None:
        requested = np.asarray(requested, dtype=float)
        result["above_request_rate"] = np.mean(y_pred > requested)
    return result

def apply_policy(raw_prediction, requested, property_price=None, max_ltv=0.90):
    prediction = np.maximum(np.asarray(raw_prediction, dtype=float), 0)
    prediction = np.minimum(prediction, np.asarray(requested, dtype=float))
    if property_price is not None:
        valid_property = np.asarray(property_price, dtype=float)
        cap = np.where(np.isfinite(valid_property) & (valid_property > 0), valid_property * max_ltv, np.inf)
        prediction = np.minimum(prediction, cap)
    return prediction
'''


def build_eda() -> None:
    cells = [
        md('''
# 01 — Phân tích khám phá dữ liệu

**Bài toán:** dự đoán `Loan Sanction Amount (USD)` cho hồ sơ đã qua bước đủ điều kiện vay.

Notebook này không giả định trước target sạch. Mọi quyết định loại dữ liệu, chọn phạm vi regression và đánh giá policy lịch sử đều dựa trên kết quả tính trực tiếp bên dưới.
'''),
        code(COMMON_SETUP),
        md("## 1. Cấu trúc và chất lượng dữ liệu"),
        code(r'''
quality = pd.DataFrame({
    "dtype": df_raw.dtypes.astype(str),
    "missing": df_raw.isna().sum(),
    "missing_pct": df_raw.isna().mean() * 100,
    "n_unique": df_raw.nunique(dropna=False),
}).sort_values(["missing_pct", "n_unique"], ascending=[False, False])
display(quality)

print(f"Số Customer ID trùng: {df_raw['Customer ID'].duplicated().sum():,}")
print(f"Số dòng trùng hoàn toàn: {df_raw.duplicated().sum():,}")
print(f"Target có trong test ngoài: {TARGET in df_external.columns}")
'''),
        code(r'''
numeric_cols = df_raw.select_dtypes(include=np.number).columns
sentinel_report = pd.DataFrame({
    "negative": [(df_raw[c] < 0).sum() for c in numeric_cols],
    "minus_999": [(df_raw[c] == -999).sum() for c in numeric_cols],
    "zero": [(df_raw[c] == 0).sum() for c in numeric_cols],
}, index=numeric_cols)
display(sentinel_report[(sentinel_report != 0).any(axis=1)])
'''),
        md("## 2. Audit target và xác định population cho regression"),
        code(r'''
y = pd.to_numeric(df_raw[TARGET], errors="coerce")
target_state = pd.Series(
    np.select(
        [y.isna(), y.eq(-999), y.eq(0), y.gt(0)],
        ["Thiếu", "Sentinel -999", "Bằng 0", "Dương"],
        default="Âm khác",
    ),
    name="Trạng thái target",
)
state_table = target_state.value_counts().to_frame("Số hồ sơ")
state_table["Tỷ lệ"] = state_table["Số hồ sơ"] / len(df_raw)
display(state_table)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
state_table["Số hồ sơ"].plot.bar(ax=axes[0], color="#355070")
axes[0].set_title("Trạng thái của target")
axes[0].set_ylabel("Số hồ sơ")

positive_target = y[y > 0]
sns.histplot(positive_target, bins=50, kde=True, ax=axes[1], color="#2a9d8f")
axes[1].set_title("Phân phối số tiền phê duyệt dương")
axes[1].set_xlabel("USD")
plt.tight_layout()
plt.show()

display(positive_target.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).to_frame("Giá trị"))
'''),
        code(r'''
n_positive = int((y > 0).sum())
n_zero = int((y == 0).sum())
n_invalid = int(y.isna().sum() + (y < 0).sum())
display(Markdown(f"""
### Kết luận phạm vi

- Có **{n_positive:,}** hồ sơ mang số tiền phê duyệt dương và phù hợp với regression.
- Có **{n_zero:,}** hồ sơ bằng 0. Không có data dictionary đủ mạnh để khẳng định đây là hồ sơ bị từ chối, nhưng chúng không thuộc cùng cơ chế sinh target với nhóm dương.
- Có **{n_invalid:,}** target thiếu hoặc âm; `-999` thể hiện sentinel chứ không phải số tiền thực.
- Để giữ bài toán thuần regression, mô hình chính chỉ học trên hồ sơ có target dương. Hệ thống vì vậy áp dụng **sau bước pre-qualification**, không quyết định duyệt hay từ chối.
"""))
'''),
        md("## 3. Số tiền yêu cầu và quy tắc phê duyệt lịch sử"),
        code(r'''
positive = df_raw.loc[y > 0].copy()
positive["SanctionRatio"] = positive[TARGET] / positive[REQUEST]
positive["RatioRounded"] = positive["SanctionRatio"].round(2)

ratio_table = positive["RatioRounded"].value_counts().sort_index().to_frame("Số hồ sơ")
ratio_table["Tỷ lệ hồ sơ"] = ratio_table["Số hồ sơ"] / len(positive)
display(ratio_table)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sample = positive.sample(min(5000, len(positive)), random_state=RANDOM_STATE)
sns.scatterplot(data=sample, x=REQUEST, y=TARGET, hue="RatioRounded", palette="viridis", alpha=.55, ax=axes[0], legend=False)
limit = max(sample[REQUEST].max(), sample[TARGET].max())
axes[0].plot([0, limit], [0, limit], "--", color="black", linewidth=1)
axes[0].set_title("Số tiền yêu cầu và số tiền phê duyệt")

ratio_table["Tỷ lệ hồ sơ"].plot.bar(ax=axes[1], color="#e76f51")
axes[1].set_title("Tỷ lệ phê duyệt / yêu cầu")
axes[1].set_ylabel("Tỷ lệ hồ sơ")
plt.tight_layout()
plt.show()
'''),
        code(r'''
top_ratios = ratio_table.sort_values("Số hồ sơ", ascending=False).head(4)
share_top = top_ratios["Tỷ lệ hồ sơ"].sum()
baseline_70_mae = np.mean(np.abs(positive[TARGET] - 0.70 * positive[REQUEST]))
target_request_corr = positive[[TARGET, REQUEST]].corr().iloc[0, 1]
display(Markdown(f"""
### Nhận xét về policy lịch sử

- Bốn tỷ lệ phổ biến nhất là {', '.join(f'{idx:.0%}' for idx in top_ratios.index)} và bao phủ **{share_top:.1%}** nhóm target dương.
- Tương quan giữa số tiền yêu cầu và số tiền phê duyệt là **{target_request_corr:.3f}**.
- Chỉ dùng quy tắc `70% × số tiền yêu cầu` đã có MAE **{baseline_70_mae:,.0f} USD** trên toàn bộ nhóm dương.
- Vì target có cấu trúc policy rời rạc, baseline 70% là đối thủ thực tế bắt buộc. Một mô hình phức tạp không vượt baseline này sẽ không có giá trị triển khai.
"""))
'''),
        md("## 4. Quan hệ với hồ sơ tài chính và tín dụng"),
        code(r'''
analysis_cols = [
    "Income (USD)", "Current Loan Expenses (USD)", "Credit Score",
    "No. of Defaults", "Property Age", "Property Price", REQUEST, TARGET,
]
corr = positive[analysis_cols].replace(-999, np.nan).corr(method="spearman")
plt.figure(figsize=(11, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
plt.title("Tương quan Spearman trên nhóm target dương")
plt.tight_layout()
plt.show()

ratio_profile = positive.groupby("RatioRounded").agg(
    n=(TARGET, "size"),
    credit_score=("Credit Score", "mean"),
    income=("Income (USD)", "mean"),
    requested=(REQUEST, "mean"),
    property_price=("Property Price", "mean"),
    sanctioned=(TARGET, "mean"),
).sort_index()
display(ratio_profile)
'''),
        code(r'''
categorical = ["Income Stability", "Profession", "Type of Employment", "Location", "Property Location"]
profiles = []
for col in categorical:
    tmp = positive.groupby(col, dropna=False).agg(
        n=(TARGET, "size"),
        mean_target=(TARGET, "mean"),
        median_ratio=("SanctionRatio", "median"),
    ).reset_index()
    tmp.insert(0, "feature", col)
    tmp = tmp.rename(columns={col: "group"})
    profiles.append(tmp)
profile_table = pd.concat(profiles, ignore_index=True)
display(profile_table.sort_values(["feature", "n"], ascending=[True, False]))
'''),
        md("## 5. So sánh phân phối train và test ngoài"),
        code(r'''
from scipy.stats import ks_2samp

shared_numeric = [c for c in df_external.select_dtypes(include=np.number).columns if c in df_raw.columns]
drift_rows = []
for col in shared_numeric:
    a = df_raw[col].replace(-999, np.nan).dropna()
    b = df_external[col].replace(-999, np.nan).dropna()
    stat, pvalue = ks_2samp(a, b)
    drift_rows.append({"feature": col, "ks_stat": stat, "p_value": pvalue, "train_mean": a.mean(), "test_mean": b.mean()})
drift = pd.DataFrame(drift_rows).sort_values("ks_stat", ascending=False)
display(drift)

max_ks = drift.iloc[0]
display(Markdown(f"Feature lệch nhất theo KS là **{max_ks['feature']}** với KS={max_ks['ks_stat']:.3f}. Giá trị nhỏ cho thấy train và test ngoài nhìn chung được sinh từ phân phối tương tự; p-value không được dùng đơn độc vì cỡ mẫu lớn."))
'''),
        md("## 6. Kết luận EDA và giả thuyết cho mô hình"),
        code(r'''
missing_top = quality[quality["missing_pct"] > 0].head(5)
display(Markdown(f"""
### Kết luận kỹ thuật

1. Population chính có {len(positive):,} hồ sơ target dương; target 0 và sentinel không được trộn vào regressor.
2. Target phụ thuộc rất mạnh vào số tiền yêu cầu và một số tỷ lệ policy rời rạc. Do đó phải so với rule baseline, không chỉ `DummyRegressor`.
3. Các cột thiếu nhiều nhất gồm {', '.join(missing_top.index.astype(str))}; imputation phải fit chỉ trên train.
4. `Customer ID`, `Name`, `Property ID` là định danh. `Gender` chỉ dùng audit fairness, không dùng production model mặc định.
5. `-999` xuất hiện cả ở target và feature; cần chuẩn hóa thành missing trước feature engineering.

### Hàm ý business

- LendScope là công cụ ước lượng **số tiền phê duyệt lịch sử cho hồ sơ đã pre-qualified**, không phải mô hình duyệt/từ chối.
- Nếu mô hình tốt chủ yếu nhờ `Loan Amount Request`, báo cáo phải nói rõ nó đang tái tạo policy lịch sử.
- Kết quả cần có policy cap và cảnh báo ngoài miền dữ liệu trước khi phục vụ qua API.
"""))
'''),
    ]
    write_notebook("01_EDA.ipynb", cells)


def build_feature_engineering() -> None:
    cells = [
        md('''
# 02 — Tiền xử lý và Feature Engineering

Mục tiêu là tạo feature có thể sử dụng tại thời điểm ra quyết định, đồng thời chứng minh mọi phép fit thống kê chỉ học từ train.
'''),
        code(COMMON_SETUP),
        code(FEATURE_FUNCTIONS),
        code(SPLIT_FUNCTIONS),
        md("## 1. Làm sạch target và chia dữ liệu trước khi fit"),
        code(r'''
train_df, val_df, test_df = make_splits(df_raw)
print(f"Train: {len(train_df):,} | Validation: {len(val_df):,} | Test nội bộ: {len(test_df):,}")
print(f"Tổng: {len(train_df) + len(val_df) + len(test_df):,}")

split_summary = pd.DataFrame({
    "train": train_df[TARGET].describe(),
    "validation": val_df[TARGET].describe(),
    "test": test_df[TARGET].describe(),
})
display(split_summary)
'''),
        md("## 2. Chuẩn hóa sentinel và xây feature business"),
        code(r'''
X_train = modeling_frame(train_df)
X_val = modeling_frame(val_df)
X_test = modeling_frame(test_df)
y_train, y_val, y_test = train_df[TARGET], val_df[TARGET], test_df[TARGET]

print(f"Số feature sau engineering: {X_train.shape[1]}")
print("Các feature mới:")
engineered = [c for c in X_train.columns if c not in df_raw.columns]
print(engineered)

assert not np.isinf(X_train.select_dtypes(include=np.number)).any().any()
assert TARGET not in X_train.columns
assert not set(DROP_COLUMNS).intersection(X_train.columns)
print("Kiểm tra leakage, identifier và giá trị vô cực: đạt")
'''),
        code(r'''
feature_quality = pd.DataFrame({
    "dtype": X_train.dtypes.astype(str),
    "missing_train": X_train.isna().sum(),
    "missing_val": X_val.isna().sum(),
    "missing_test": X_test.isna().sum(),
    "n_unique_train": X_train.nunique(dropna=False),
}).sort_values("missing_train", ascending=False)
display(feature_quality)
'''),
        md("## 3. Fit preprocessing chỉ trên train"),
        code(MODEL_FUNCTIONS),
        code(r'''
preprocessor = build_preprocessor(X_train, scale_numeric=True)
Xt_train = preprocessor.fit_transform(X_train)
Xt_val = preprocessor.transform(X_val)
Xt_test = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()

print(f"Ma trận train sau preprocessing: {Xt_train.shape}")
print(f"Ma trận validation: {Xt_val.shape}")
print(f"Ma trận test: {Xt_test.shape}")
print(f"NaN sau preprocessing: {np.isnan(Xt_train).sum():,}")
print("Preprocessor chỉ fit trên X_train; validation/test chỉ gọi transform.")
'''),
        md("## 4. Kiểm tra sức giải thích và đa cộng tuyến"),
        code(r'''
numeric_engineered = X_train.select_dtypes(include=np.number).copy()
numeric_engineered[TARGET] = y_train.values
corr_target = numeric_engineered.corr(method="spearman")[TARGET].drop(TARGET).sort_values(key=np.abs, ascending=False)
display(corr_target.to_frame("Spearman với target"))

top_numeric = corr_target.head(12).index
plt.figure(figsize=(11, 8))
sns.heatmap(numeric_engineered[list(top_numeric)].corr(method="spearman"), cmap="vlag", center=0)
plt.title("Tương quan giữa các numeric feature nổi bật")
plt.tight_layout()
plt.show()
'''),
        code(r'''
from sklearn.feature_selection import mutual_info_regression

numeric_imputed = pd.DataFrame(
    SimpleImputer(strategy="median").fit_transform(X_train.select_dtypes(include=np.number)),
    columns=X_train.select_dtypes(include=np.number).columns,
    index=X_train.index,
)
mi = pd.Series(
    mutual_info_regression(numeric_imputed, y_train, random_state=RANDOM_STATE),
    index=numeric_imputed.columns,
).sort_values(ascending=False)
display(mi.to_frame("Mutual information"))
'''),
        md("## 5. Kết luận feature engineering"),
        code(r'''
top_corr = corr_target.index[0]
top_mi = mi.index[0]
display(Markdown(f"""
### Kết luận

- Dữ liệu được chia 70% train, 15% validation và 15% test nội bộ trước mọi phép fit.
- Feature có tương quan đơn biến mạnh nhất với target là **{top_corr}**; feature có mutual information cao nhất là **{top_mi}**.
- `Loan Amount Request (USD)` và các tỷ lệ dẫn xuất từ nó nhiều khả năng chi phối mô hình. Đây là tín hiệu đúng nghiệp vụ nhưng cũng cho thấy model đang học policy lịch sử.
- Identifier và `Gender` đã bị loại khỏi production feature set. `Gender` sẽ quay lại ở notebook diễn giải chỉ để audit sai số.
- Pipeline imputation và encoding xử lý được missing/unseen categories mà không học thông tin từ validation hoặc test.
"""))
'''),
    ]
    write_notebook("02_Feature_Engineering.ipynb", cells)


def build_experiments() -> None:
    cells = [
        md('''
# 03 — Thực nghiệm mô hình Regression

Mỗi mô hình phải vượt hai mốc: baseline thống kê và baseline policy `70% × số tiền yêu cầu`. Validation dùng để chọn mô hình; test nội bộ chỉ mở một lần sau khi đã chọn.
'''),
        code(COMMON_SETUP),
        code(FEATURE_FUNCTIONS),
        code(SPLIT_FUNCTIONS),
        code(MODEL_FUNCTIONS),
        md("## 1. Chuẩn bị split cố định"),
        code(r'''
train_df, val_df, test_df = make_splits(df_raw)
X_train, X_val, X_test = map(modeling_frame, [train_df, val_df, test_df])
y_train, y_val, y_test = train_df[TARGET], val_df[TARGET], test_df[TARGET]
print(X_train.shape, X_val.shape, X_test.shape)
'''),
        md("## 2. Baseline thống kê và policy"),
        code(r'''
from sklearn.dummy import DummyRegressor

results = []
fitted = {}

dummy = DummyRegressor(strategy="median").fit(X_train, y_train)
dummy_pred = dummy.predict(X_val)
results.append({"model": "DummyMedian", **regression_metrics(y_val, dummy_pred, val_df[REQUEST])})

rule70_pred = 0.70 * val_df[REQUEST].to_numpy()
results.append({"model": "Rule70Percent", **regression_metrics(y_val, rule70_pred, val_df[REQUEST])})
display(pd.DataFrame(results).set_index("model"))
'''),
        md("## 3. Huấn luyện các họ mô hình"),
        code(r'''
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import time

model_specs = {
    "Ridge": (Ridge(alpha=10.0), True),
    "RandomForest": (RandomForestRegressor(
        n_estimators=260, max_depth=18, min_samples_leaf=3,
        max_features=0.8, n_jobs=-1, random_state=RANDOM_STATE,
    ), False),
    "HistGradientBoosting": (HistGradientBoostingRegressor(
        learning_rate=0.06, max_iter=350, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=RANDOM_STATE,
    ), False),
    "XGBoost": (XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.04,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
        objective="reg:squarederror", n_jobs=-1, random_state=RANDOM_STATE,
    ), False),
    "LightGBM": (LGBMRegressor(
        n_estimators=500, learning_rate=0.04, num_leaves=31,
        max_depth=-1, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=2.0, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
    ), False),
}

for name, (estimator, scale_numeric) in model_specs.items():
    started = time.perf_counter()
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X_train, scale_numeric=scale_numeric)),
        ("model", estimator),
    ])
    pipeline.fit(X_train, y_train)
    raw_pred = pipeline.predict(X_val)
    policy_pred = apply_policy(raw_pred, val_df[REQUEST], val_df["Property Price"])
    row = {
        "model": name,
        **regression_metrics(y_val, policy_pred, val_df[REQUEST]),
        "fit_seconds": time.perf_counter() - started,
        "raw_negative_rate": np.mean(raw_pred < 0),
    }
    results.append(row)
    fitted[name] = pipeline
    print(f"{name}: MAE={row['mae']:,.2f}, R2={row['r2']:.4f}, thời gian={row['fit_seconds']:.1f}s")

comparison = pd.DataFrame(results).sort_values("mae").reset_index(drop=True)
display(comparison)
'''),
        md("## 4. Cross-validation cho các ứng viên tốt nhất"),
        code(r'''
from sklearn.model_selection import KFold, cross_val_score

candidate_names = comparison[comparison["model"].isin(fitted)].head(3)["model"].tolist()
cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
cv_rows = []
for name in candidate_names:
    scores = -cross_val_score(
        fitted[name], X_train, y_train,
        scoring="neg_mean_absolute_error", cv=cv, n_jobs=1,
    )
    cv_rows.append({
        "model": name,
        "cv_mae_mean": scores.mean(),
        "cv_mae_std": scores.std(),
        "folds": scores.tolist(),
    })
    print(f"{name}: CV MAE={scores.mean():,.2f} ± {scores.std():,.2f}")
cv_table = pd.DataFrame(cv_rows).sort_values("cv_mae_mean")
display(cv_table)
'''),
        md("## 5. Chọn mô hình bằng cross-validation và mở test đúng một lần"),
        code(r'''
best_name = cv_table.iloc[0]["model"]
print(f"Mô hình được chọn theo CV MAE trước khi mở test: {best_name}")

train_val_df = pd.concat([train_df, val_df]).sort_index()
X_train_val = modeling_frame(train_val_df)
y_train_val = train_val_df[TARGET]
best_estimator, scale_numeric = model_specs[best_name]
from sklearn.base import clone
final_pipeline = Pipeline([
    ("preprocessor", build_preprocessor(X_train_val, scale_numeric=scale_numeric)),
    ("model", clone(best_estimator)),
])
final_pipeline.fit(X_train_val, y_train_val)

raw_test_pred = final_pipeline.predict(X_test)
test_pred = apply_policy(raw_test_pred, test_df[REQUEST], test_df["Property Price"])
test_metrics = regression_metrics(y_test, test_pred, test_df[REQUEST])
rule70_test = regression_metrics(y_test, 0.70 * test_df[REQUEST], test_df[REQUEST])
test_compare = pd.DataFrame([test_metrics, rule70_test], index=[best_name, "Rule70Percent"])
display(test_compare)

# Calibration dùng model chỉ fit trên train và residual của validation.
calibration_raw = fitted[best_name].predict(X_val)
calibration_pred = apply_policy(calibration_raw, val_df[REQUEST], val_df["Property Price"])
calibration_abs_error = np.abs(y_val.to_numpy() - calibration_pred)
calibration_q90 = float(np.quantile(calibration_abs_error, 0.90))
calibration_lower = np.maximum(calibration_pred - calibration_q90, 0)
calibration_upper = np.minimum(calibration_pred + calibration_q90, val_df[REQUEST].to_numpy())
calibration_relative_width = (calibration_upper - calibration_lower) / np.maximum(calibration_pred, 1)
manual_review_width_threshold = float(np.quantile(calibration_relative_width, 0.90))
print(f"Calibration absolute-error P90: {calibration_q90:,.2f} USD")
print(f"Ngưỡng relative interval width P90: {manual_review_width_threshold:.3f}")
'''),
        code(r'''
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].scatter(y_test, test_pred, alpha=.35, s=14)
limit = max(y_test.max(), test_pred.max())
axes[0].plot([0, limit], [0, limit], "--", color="black")
axes[0].set(title="Thực tế và dự đoán", xlabel="Thực tế", ylabel="Dự đoán")

residual = test_pred - y_test.to_numpy()
sns.histplot(residual, bins=50, kde=True, ax=axes[1])
axes[1].axvline(0, color="black", linestyle="--")
axes[1].set_title("Phân phối residual")

axes[2].scatter(test_pred, residual, alpha=.35, s=14)
axes[2].axhline(0, color="black", linestyle="--")
axes[2].set(title="Residual theo dự đoán", xlabel="Dự đoán", ylabel="Dự đoán - thực tế")
plt.tight_layout()
plt.show()
'''),
        md("## 6. Lưu artifact và theo dõi thí nghiệm"),
        code(r'''
import joblib
import mlflow
from datetime import datetime, timezone

artifact_dir = ROOT / "outputs" / "notebook_artifacts"
report_dir = ROOT / "outputs" / "reports"
artifact_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)

reference = {}
for col in X_train_val.select_dtypes(include=np.number).columns:
    values = X_train_val[col].replace([np.inf, -np.inf], np.nan).dropna()
    reference[col] = {
        "p01": float(values.quantile(.01)), "p50": float(values.quantile(.50)),
        "p99": float(values.quantile(.99)), "mean": float(values.mean()),
    }

artifact = {
    "pipeline": final_pipeline,
    "model_name": best_name,
    "target": TARGET,
    "feature_columns": X_train_val.columns.tolist(),
    "metrics": {k: float(v) for k, v in test_metrics.items()},
    "calibration_abs_error_q90": calibration_q90,
    "manual_review_width_threshold": manual_review_width_threshold,
    "reference": reference,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "scope": "positive_sanction_prequalified_applications",
}
joblib.dump(artifact, artifact_dir / "best_model.joblib")

predictions = test_df.copy()
predictions["raw_prediction"] = raw_test_pred
predictions["prediction"] = test_pred
predictions["residual"] = test_pred - predictions[TARGET]
predictions["absolute_error"] = predictions["residual"].abs()
predictions.to_csv(report_dir / "internal_test_predictions.csv", index=True, index_label="source_index")
comparison.to_csv(report_dir / "model_comparison.csv", index=False)

with open(report_dir / "test_metrics.json", "w", encoding="utf-8") as f:
    json.dump({"model": best_name, **artifact["metrics"]}, f, ensure_ascii=False, indent=2)

tracking_db = (ROOT / "outputs" / "mlflow.db").resolve()
mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
mlflow.set_experiment("lendscope-regression")
with mlflow.start_run(run_name=f"notebook-{best_name}"):
    mlflow.log_param("model", best_name)
    mlflow.log_param("population", artifact["scope"])
    mlflow.log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items()})
    mlflow.log_artifact(str(report_dir / "model_comparison.csv"))
print(f"Đã lưu artifact: {(artifact_dir / 'best_model.joblib').relative_to(ROOT)}")
'''),
        md("## 7. Kết luận thực nghiệm"),
        code(r'''
mae_gain = 1 - test_metrics["mae"] / rule70_test["mae"]
display(Markdown(f"""
### Kết luận

- Ba mô hình dẫn đầu validation được kiểm tra bằng 3-fold CV; mô hình có CV MAE tốt nhất và được chọn trước khi mở test là **{best_name}**.
- Trên test nội bộ khóa trước, MAE đạt **{test_metrics['mae']:,.0f} USD**, RMSE **{test_metrics['rmse']:,.0f} USD**, R² **{test_metrics['r2']:.4f}**.
- Tỷ lệ dự đoán nằm trong ±10% là **{test_metrics['within_10pct']:.1%}**, trong ±20% là **{test_metrics['within_20pct']:.1%}**.
- So với rule 70%, MAE cải thiện **{mae_gain:.1%}**. Đây mới là mức tăng có ý nghĩa vì rule 70% phản ánh policy thật trong dữ liệu.
- `mean_signed_error={test_metrics['mean_signed_error']:,.0f}` USD cho biết xu hướng lệch tổng thể; dấu dương là thiên về cấp cao hơn lịch sử, dấu âm là thấp hơn.
- Test ngoài không có nhãn nên không được dùng để công bố metrics.
"""))
'''),
    ]
    write_notebook("03_Model_Experiments.ipynb", cells)


def build_interpretation() -> None:
    cells = [
        md('''
# 04 — Diễn giải mô hình và phân tích lỗi

Notebook này đọc đúng artifact và test prediction đã khóa ở notebook 03. Mục tiêu không chỉ là xếp hạng feature mà còn xác định mô hình sai ở đâu và liệu sai số có bất cân xứng giữa các nhóm hay không.
'''),
        code(COMMON_SETUP),
        code(FEATURE_FUNCTIONS),
        code(MODEL_FUNCTIONS),
        code(r'''
import joblib

artifact_path = ROOT / "outputs" / "notebook_artifacts" / "best_model.joblib"
prediction_path = ROOT / "outputs" / "reports" / "internal_test_predictions.csv"
artifact = joblib.load(artifact_path)
predictions = pd.read_csv(prediction_path, index_col="source_index")
pipeline = artifact["pipeline"]
model_name = artifact["model_name"]
print(f"Model: {model_name} | Test rows: {len(predictions):,}")
'''),
        md("## 1. Permutation importance trên dữ liệu gốc"),
        code(r'''
from sklearn.inspection import permutation_importance

X_test = modeling_frame(predictions)
y_test = predictions[TARGET]
perm = permutation_importance(
    pipeline, X_test, y_test,
    scoring="neg_mean_absolute_error", n_repeats=5,
    random_state=RANDOM_STATE, n_jobs=1,
)
perm_df = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)
display(perm_df.head(20))

plt.figure(figsize=(10, 7))
top = perm_df.head(15).sort_values("importance_mean")
plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
plt.xlabel("Mức MAE tăng khi xáo trộn feature")
plt.title("Permutation importance")
plt.tight_layout()
plt.show()
'''),
        md("## 2. SHAP trên không gian sau preprocessing"),
        code(r'''
import shap

preprocessor = pipeline.named_steps["preprocessor"]
estimator = pipeline.named_steps["model"]
sample_raw = X_test.sample(min(500, len(X_test)), random_state=RANDOM_STATE)
background_raw = X_test.sample(min(150, len(X_test)), random_state=7)
sample_transformed = preprocessor.transform(sample_raw)
background_transformed = preprocessor.transform(background_raw)
feature_names = preprocessor.get_feature_names_out()

explainer = shap.TreeExplainer(
    estimator,
    data=background_transformed[:100],
    feature_names=feature_names,
    feature_perturbation="interventional",
)
shap_values = explainer(sample_transformed, check_additivity=False)
shap.plots.beeswarm(shap_values, max_display=18, show=False)
plt.title(f"SHAP summary — {model_name}")
plt.tight_layout()
plt.show()

shap_importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False)
display(shap_importance.head(20))
'''),
        code(r'''
row_position = int(np.argmax(predictions["absolute_error"].to_numpy()))
row_index = predictions.index[row_position]
row_raw = modeling_frame(predictions.loc[[row_index]])
row_transformed = preprocessor.transform(row_raw)
row_shap = explainer(row_transformed)
shap.plots.waterfall(row_shap[0], max_display=15, show=False)
plt.title("Giải thích hồ sơ có absolute error lớn nhất")
plt.tight_layout()
plt.show()

display(predictions.loc[[row_index], [REQUEST, TARGET, "prediction", "residual", "Credit Score", "Income (USD)", "Property Price"]])
'''),
        md("## 3. Residual và hiệu suất theo phân khúc"),
        code(r'''
def segment_metrics(frame, group_col):
    rows = []
    for group, part in frame.groupby(group_col, dropna=False, observed=True):
        m = regression_metrics(part[TARGET], part["prediction"], part[REQUEST])
        rows.append({"group": group, "n": len(part), **m})
    return pd.DataFrame(rows).sort_values("mae", ascending=False)

predictions["target_band"] = pd.qcut(predictions[TARGET], q=5, duplicates="drop")
predictions["credit_band"] = pd.cut(predictions["Credit Score"], bins=[0, 650, 700, 750, 800, 850, np.inf])
predictions["income_band"] = pd.qcut(predictions["Income (USD)"], q=5, duplicates="drop")

for column in ["target_band", "credit_band", "income_band", "Gender", "Location", "Income Stability"]:
    print(f"\nPhân khúc: {column}")
    display(segment_metrics(predictions, column))
'''),
        code(r'''
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.boxplot(data=predictions, x="target_band", y="residual", ax=axes[0])
axes[0].tick_params(axis="x", rotation=25)
axes[0].axhline(0, color="black", linestyle="--")
axes[0].set_title("Residual theo nhóm target")

sns.scatterplot(data=predictions, x=REQUEST, y="residual", hue="Credit Score", palette="viridis", alpha=.5, ax=axes[1])
axes[1].axhline(0, color="black", linestyle="--")
axes[1].set_title("Residual theo số tiền yêu cầu")
plt.tight_layout()
plt.show()
'''),
        md("## 4. Kết luận diễn giải"),
        code(r'''
top_perm = perm_df.iloc[0]["feature"]
top_shap = shap_importance.iloc[0]["feature"]
gender_metrics = segment_metrics(predictions, "Gender")
gender_gap = gender_metrics["mae"].max() - gender_metrics["mae"].min()
worst_band = segment_metrics(predictions, "target_band").iloc[0]
display(Markdown(f"""
### Kết luận

- Feature quan trọng nhất theo permutation là **{top_perm}**; theo SHAP là **{top_shap}**.
- Nếu các feature liên quan số tiền yêu cầu đứng đầu, mô hình chủ yếu tái tạo policy sanction theo nhu cầu vay. Điều này phù hợp dữ liệu nhưng không chứng minh quan hệ nhân quả với khả năng trả nợ.
- Nhóm target có MAE cao nhất là **{worst_band['group']}**, MAE **{worst_band['mae']:,.0f} USD** trên {int(worst_band['n']):,} hồ sơ.
- Chênh lệch MAE quan sát giữa các nhóm giới tính là **{gender_gap:,.0f} USD**. Đây là audit hiệu suất, không đủ để kết luận fairness nếu chưa kiểm soát cấu trúc hồ sơ.
- Production model không dùng `Gender`; các biến vị trí vẫn cần theo dõi vì có thể đóng vai trò proxy.
"""))
'''),
    ]
    write_notebook("04_Model_Interpretation.ipynb", cells)


def build_business_report() -> None:
    cells = [
        md('''
# 05 — Báo cáo Business cho LendScope

**Đối tượng:** nhóm vận hành tín dụng và quản lý sản phẩm.

Báo cáo chuyển metrics kỹ thuật thành tác động vận hành. Kết quả là mức tiền tham chiếu cho hồ sơ đã pre-qualified, không phải quyết định tín dụng tự động.
'''),
        code(COMMON_SETUP),
        code(MODEL_FUNCTIONS),
        code(r'''
import joblib

artifact = joblib.load(ROOT / "outputs" / "notebook_artifacts" / "best_model.joblib")
predictions = pd.read_csv(ROOT / "outputs" / "reports" / "internal_test_predictions.csv", index_col="source_index")
metrics = artifact["metrics"]
print(f"Model: {artifact['model_name']} | Holdout: {len(predictions):,} hồ sơ")
'''),
        md("## 1. KPI mô hình"),
        code(r'''
kpi = pd.DataFrame({
    "KPI": ["MAE", "RMSE", "R²", "WAPE", "Trong ±10%", "Trong ±20%", "Tỷ lệ over-sanction"],
    "Giá trị": [metrics["mae"], metrics["rmse"], metrics["r2"], metrics["wape"], metrics["within_10pct"], metrics["within_20pct"], metrics["over_rate"]],
})
display(kpi)

rule70 = 0.70 * predictions[REQUEST]
rule70_metrics = regression_metrics(predictions[TARGET], rule70, predictions[REQUEST])
comparison = pd.DataFrame({
    "LendScope": metrics,
    "Rule 70%": rule70_metrics,
}).T[["mae", "rmse", "r2", "wape", "within_10pct", "within_20pct"]]
display(comparison)
'''),
        md("## 2. Over-sanction, under-sanction và chi phí giả định"),
        code(r'''
predictions["direction"] = np.select(
    [predictions["residual"] > 0, predictions["residual"] < 0],
    ["Cao hơn lịch sử", "Thấp hơn lịch sử"],
    default="Bằng lịch sử",
)
direction_summary = predictions.groupby("direction").agg(
    n=(TARGET, "size"),
    total_gap=("residual", "sum"),
    mean_gap=("residual", "mean"),
    mean_absolute_error=("absolute_error", "mean"),
)
direction_summary["share"] = direction_summary["n"] / len(predictions)
display(direction_summary)

OVER_COST_RATE = 0.05
UNDER_OPPORTUNITY_RATE = 0.02
over_exposure = predictions["residual"].clip(lower=0).sum()
under_opportunity = (-predictions["residual"].clip(upper=0)).sum()
assumed_cost = over_exposure * OVER_COST_RATE + under_opportunity * UNDER_OPPORTUNITY_RATE
print(f"Tổng phần dự đoán cao hơn lịch sử: {over_exposure:,.0f} USD")
print(f"Tổng phần dự đoán thấp hơn lịch sử: {under_opportunity:,.0f} USD")
print(f"Chi phí giả định để so sánh kịch bản: {assumed_cost:,.0f} USD")
print("Các tỷ lệ chi phí chỉ là giả định minh họa, không phải số liệu tài chính thực tế.")
'''),
        md("## 3. Khoảng bất định và manual review"),
        code(r'''
abs_q90 = float(artifact["calibration_abs_error_q90"])
predictions["lower_90"] = np.maximum(predictions["prediction"] - abs_q90, 0)
predictions["upper_90"] = np.minimum(predictions["prediction"] + abs_q90, predictions[REQUEST])
coverage = ((predictions[TARGET] >= predictions["lower_90"]) & (predictions[TARGET] <= predictions["upper_90"])).mean()
predictions["relative_interval_width"] = (predictions["upper_90"] - predictions["lower_90"]) / predictions["prediction"].clip(lower=1)
width_threshold = float(artifact["manual_review_width_threshold"])
predictions["manual_review"] = predictions["relative_interval_width"] > width_threshold

print(f"Biên absolute error P90 học từ validation: {abs_q90:,.0f} USD")
print(f"Ngưỡng relative interval width học từ validation: {width_threshold:.3f}")
print(f"Coverage quan sát của khoảng đã cap: {coverage:.1%}")
print(f"Hồ sơ gắn cờ manual review trong holdout: {predictions['manual_review'].sum():,}")
display(predictions.nlargest(15, "absolute_error")[[REQUEST, TARGET, "prediction", "lower_90", "upper_90", "absolute_error", "Credit Score", "Income (USD)"]])
'''),
        md("## 4. Hiệu suất theo phân khúc business"),
        code(r'''
predictions["amount_segment"] = pd.qcut(predictions[TARGET], q=5, duplicates="drop")
segment = predictions.groupby("amount_segment", observed=True).agg(
    n=(TARGET, "size"),
    actual_mean=(TARGET, "mean"),
    predicted_mean=("prediction", "mean"),
    mae=("absolute_error", "mean"),
    bias=("residual", "mean"),
    manual_review_rate=("manual_review", "mean"),
)
display(segment)

fig, ax = plt.subplots(figsize=(11, 6))
segment[["actual_mean", "predicted_mean"]].plot.bar(ax=ax)
ax.set_title("Số tiền thực tế và dự đoán theo phân khúc")
ax.set_ylabel("USD")
ax.tick_params(axis="x", rotation=25)
plt.tight_layout()
plt.show()
'''),
        md("## 5. Kịch bản hồ sơ"),
        code(r'''
scenario_cols = [REQUEST, "Income (USD)", "Credit Score", "Property Price", "Current Loan Expenses (USD)", TARGET, "prediction", "lower_90", "upper_90"]
scenarios = pd.concat([
    predictions.nsmallest(1, TARGET),
    predictions.iloc[[(predictions[TARGET] - predictions[TARGET].median()).abs().argmin()]],
    predictions.nlargest(1, TARGET),
    predictions.nlargest(1, "absolute_error"),
])
scenarios.index = ["Khoản nhỏ", "Khoản trung vị", "Khoản lớn", "Sai số lớn nhất"]
display(scenarios[scenario_cols])
'''),
        md("## 6. Xuất báo cáo vận hành và kết luận"),
        code(r'''
report_dir = ROOT / "outputs" / "reports"
review_cols = [
    "Customer ID", REQUEST, TARGET, "prediction", "lower_90", "upper_90",
    "absolute_error", "Credit Score", "Income (USD)", "Property Price",
]
predictions.loc[predictions["manual_review"], review_cols].sort_values("absolute_error", ascending=False).to_csv(
    report_dir / "manual_review_holdout.csv", index=False
)

mae_gain = 1 - metrics["mae"] / rule70_metrics["mae"]
display(Markdown(f"""
### Kết luận business

1. LendScope đạt MAE **{metrics['mae']:,.0f} USD** và cải thiện **{mae_gain:.1%}** so với quy tắc 70% trên holdout.
2. **{metrics['within_20pct']:.1%}** dự đoán nằm trong ±20% so với số tiền lịch sử. Những hồ sơ ngoài vùng này cần review thay vì tự động áp dụng.
3. Mô hình tái tạo chính sách sanction lịch sử; nó không có nhãn default sau giải ngân nên không thể tuyên bố mức tiền là an toàn về rủi ro tín dụng.
4. Khoảng bất định đơn giản đạt coverage quan sát **{coverage:.1%}** sau khi cap. Đây là công cụ triage, chưa phải conformal guarantee chính thức.
5. Production phải luôn cap theo số tiền yêu cầu, policy tài sản và gắn cờ dữ liệu ngoài miền training.
6. Dataset có provenance hạn chế; kết quả phù hợp học tập và portfolio, không dùng để ra quyết định tín dụng thật.
"""))
'''),
    ]
    write_notebook("05_Business_Report.ipynb", cells)


if __name__ == "__main__":
    NB_DIR.mkdir(parents=True, exist_ok=True)
    build_eda()
    build_feature_engineering()
    build_experiments()
    build_interpretation()
    build_business_report()
    print("Created 5 LendScope notebooks.")

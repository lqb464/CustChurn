"""Dashboard hỗ trợ rà soát hạn mức vay sau bước tiền thẩm định."""

import io
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("LENDSCOPE_API_URL", "http://localhost:8000")

NUMERIC_DEFAULTS = {
    "Age": 40,
    "Income (USD)": 2_222.44,
    "Loan Amount Request (USD)": 75_128.08,
    "Current Loan Expenses (USD)": 375.21,
    "Dependents": 2,
    "Credit Score": 739.82,
    "No. of Defaults": 0,
    "Property Age": 2_223.25,
    "Property Type": 2,
    "Co-Applicant": 1,
    "Property Price": 109_993.61,
}

CATEGORY_OPTIONS = {
    "Income Stability": ["Low", "High"],
    "Profession": [
        "Working", "Commercial associate", "Pensioner", "State servant",
        "Unemployed", "Businessman", "Student", "Maternity leave",
    ],
    "Type of Employment": [
        "Laborers", "Sales staff", "Core staff", "Managers", "Drivers",
        "Accountants", "High skill tech staff", "Medicine staff", "Security staff",
    ],
    "Location": ["Semi-Urban", "Rural", "Urban"],
    "Expense Type 1": ["N", "Y"],
    "Expense Type 2": ["Y", "N"],
    "Has Active Credit Card": ["Active", "Inactive", "Unpossessed"],
    "Property Location": ["Semi-Urban", "Rural", "Urban"],
}


def api_model() -> dict[str, Any]:
    response = requests.get(f"{API_URL}/model", timeout=10)
    response.raise_for_status()
    return response.json()


def api_predict(records: list[dict[str, Any]], batch_size: int = 1000) -> pd.DataFrame:
    """Tự chia lô theo giới hạn API để xử lý CSV lớn ổn định."""
    predictions: list[pd.DataFrame] = []
    for start in range(0, len(records), batch_size):
        response = requests.post(
            f"{API_URL}/predict",
            json={"records": records[start : start + batch_size]},
            timeout=120,
        )
        response.raise_for_status()
        predictions.append(pd.DataFrame(response.json()["predictions"]))
    return pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")


def money(value: float) -> str:
    return f"{value:,.0f} USD"


def render_prediction_summary(prediction: pd.Series, request: float, property_price: float) -> None:
    sanctioned = float(prediction["predicted_sanction_amount_usd"])
    lower = float(prediction["prediction_lower_90_usd"])
    upper = float(prediction["prediction_upper_90_usd"])
    request_ratio = sanctioned / request if request else 0.0
    ltv = sanctioned / property_price if property_price else 0.0

    st.markdown("### Kết quả đề xuất")
    first, second, third, fourth = st.columns(4)
    first.metric("Hạn mức ước lượng", money(sanctioned))
    second.metric("Khoảng dự báo 90%", f"{money(lower)} – {money(upper)}")
    third.metric("Tỷ lệ trên yêu cầu", f"{request_ratio:.1%}")
    fourth.metric("LTV ước lượng", f"{ltv:.1%}")

    comparison = pd.DataFrame(
        {
            "Giá trị": [request, sanctioned, property_price * 0.8],
        },
        index=["Số tiền yêu cầu", "Hạn mức ước lượng", "Trần 80% tài sản"],
    )
    st.bar_chart(comparison, horizontal=True, color="#1f6f5f")

    review = bool(prediction["manual_review"])
    warning = str(prediction.get("input_warning", "")).strip()
    if review:
        st.warning("Hồ sơ cần rà soát thủ công vì khoảng dự báo tương đối rộng.")
    else:
        st.success("Độ bất định nằm trong vùng chấp nhận theo ngưỡng validation.")
    if warning:
        st.error(f"Cảnh báo chất lượng đầu vào: {warning}")


st.set_page_config(page_title="LendScope Decision Support", page_icon=None, layout="wide")
st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(180deg, #f5f8f7 0%, #ffffff 38%);}
    .stApp, .stApp p, .stApp label, .stApp h2, .stApp h3 {color: #173b34;}
    [data-testid="stSidebar"] {background: #102a26;}
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #edf8f5;
    }
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: #173b34; border-color: #315c52;
    }
    .hero {padding: 2.2rem 2.4rem; border-radius: 18px; color: #f7fffc;
           background: linear-gradient(120deg, #123c35 0%, #1f6f5f 68%, #56a38f 100%);
           box-shadow: 0 12px 30px rgba(18,60,53,.18); margin-bottom: 1.3rem;}
    .hero h1 {font-size: 2.55rem; margin: 0 0 .4rem 0; color: #ffffff;}
    .hero p {font-size: 1.05rem; margin: 0; max-width: 780px; color: #dff5ee;}
    .scope {padding: .9rem 1rem; border-left: 4px solid #cf8d32; background: #fff8eb;
            border-radius: 8px; margin: .8rem 0 1.2rem 0;}
    [data-testid="stMetric"] {background: #ffffff; border: 1px solid #dbe9e5;
                              padding: 1rem; border-radius: 12px;}
    </style>
    <div class="hero">
      <h1>LendScope Decision Support</h1>
      <p>Ước lượng khoản vay được phê duyệt, lượng hóa độ bất định và ưu tiên hồ sơ cần chuyên viên rà soát.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model = api_model()
    api_ready = True
except (requests.RequestException, KeyError, ValueError):
    model = {"model_name": "Chưa sẵn sàng", "model_version": "-", "metrics": {}}
    api_ready = False

with st.sidebar:
    st.markdown("## Trạng thái mô hình")
    status_text = "API sẵn sàng" if api_ready else "Chưa kết nối API"
    if api_ready:
        st.success(status_text)
    else:
        st.error(status_text)
    st.write(f"Mô hình: {model.get('model_name', '-')}")
    st.write(f"Phiên bản: {model.get('model_version', '-')}")
    metrics = model.get("metrics", {})
    st.metric("MAE holdout", money(float(metrics.get("mae", 0))))
    st.metric("R² holdout", f"{float(metrics.get('r2', 0)):.4f}")
    st.metric("Coverage khoảng 90%", f"{float(metrics.get('prediction_interval_90_coverage', 0)):.1%}")
    st.markdown("---")
    st.caption("Population: hồ sơ đã qua bước tiền thẩm định và thuộc phạm vi regression.")

st.markdown(
    '<div class="scope"><strong>Phạm vi sử dụng:</strong> Công cụ hỗ trợ đề xuất hạn mức sau tiền thẩm định. '
    'Không dự đoán eligibility, xác suất vỡ nợ hoặc thay thế quyết định của chuyên viên tín dụng.</div>',
    unsafe_allow_html=True,
)

single_tab, batch_tab, governance_tab = st.tabs(
    ["Đánh giá một hồ sơ", "Phân tích danh mục CSV", "Mô hình và quản trị"]
)

with single_tab:
    st.markdown("### Thông tin hồ sơ")
    st.caption("Giá trị mặc định phản ánh trung vị hoặc nhóm phổ biến trong tập huấn luyện.")
    with st.form("single_application"):
        left, middle, right = st.columns(3)
        with left:
            age = st.number_input("Tuổi khách hàng", 18, 100, int(NUMERIC_DEFAULTS["Age"]))
            income = st.number_input("Thu nhập (USD)", 0.0, value=NUMERIC_DEFAULTS["Income (USD)"], step=100.0)
            income_stability = st.selectbox("Độ ổn định thu nhập", CATEGORY_OPTIONS["Income Stability"])
            profession = st.selectbox("Nghề nghiệp", CATEGORY_OPTIONS["Profession"])
            employment = st.selectbox("Loại công việc", CATEGORY_OPTIONS["Type of Employment"])
            location = st.selectbox("Khu vực cư trú", CATEGORY_OPTIONS["Location"])
        with middle:
            request_amount = st.number_input(
                "Số tiền yêu cầu (USD)", 0.0, value=NUMERIC_DEFAULTS["Loan Amount Request (USD)"], step=1_000.0
            )
            current_expense = st.number_input(
                "Chi phí khoản vay hiện tại (USD)", 0.0,
                value=NUMERIC_DEFAULTS["Current Loan Expenses (USD)"], step=25.0,
            )
            expense_1 = st.selectbox("Nhóm chi phí 1", CATEGORY_OPTIONS["Expense Type 1"])
            expense_2 = st.selectbox("Nhóm chi phí 2", CATEGORY_OPTIONS["Expense Type 2"])
            dependents = st.number_input("Số người phụ thuộc", 0, 20, int(NUMERIC_DEFAULTS["Dependents"]))
            credit_score = st.number_input("Điểm tín dụng", 0.0, 1_000.0, NUMERIC_DEFAULTS["Credit Score"], 5.0)
        with right:
            defaults = st.number_input("Số lần vỡ nợ", 0, 20, int(NUMERIC_DEFAULTS["No. of Defaults"]))
            active_card = st.selectbox("Trạng thái thẻ tín dụng", CATEGORY_OPTIONS["Has Active Credit Card"])
            property_age = st.number_input(
                "Property Age theo schema nguồn", 0.0, value=NUMERIC_DEFAULTS["Property Age"], step=100.0,
                help="Cột nguồn có quan hệ gần như trùng với Income; xem phần quản trị dữ liệu.",
            )
            property_type = st.number_input("Loại tài sản", 1, 4, int(NUMERIC_DEFAULTS["Property Type"]))
            property_location = st.selectbox("Khu vực tài sản", CATEGORY_OPTIONS["Property Location"])
            co_applicant = st.number_input("Số đồng người vay", 0, 10, int(NUMERIC_DEFAULTS["Co-Applicant"]))
            property_price = st.number_input(
                "Giá trị tài sản (USD)", 0.0, value=NUMERIC_DEFAULTS["Property Price"], step=1_000.0
            )
        submitted = st.form_submit_button("Ước lượng hạn mức", type="primary", use_container_width=True)

    if submitted:
        record = {
            "Age": age, "Income (USD)": income, "Income Stability": income_stability,
            "Profession": profession, "Type of Employment": employment, "Location": location,
            "Loan Amount Request (USD)": request_amount,
            "Current Loan Expenses (USD)": current_expense, "Expense Type 1": expense_1,
            "Expense Type 2": expense_2, "Dependents": dependents, "Credit Score": credit_score,
            "No. of Defaults": defaults, "Has Active Credit Card": active_card,
            "Property Age": property_age, "Property Type": property_type,
            "Property Location": property_location, "Co-Applicant": co_applicant,
            "Property Price": property_price,
        }
        try:
            st.session_state["single_prediction"] = api_predict([record]).iloc[0]
            st.session_state["single_request"] = request_amount
            st.session_state["single_property"] = property_price
        except requests.RequestException as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"Dự báo thất bại: {detail}")

    if "single_prediction" in st.session_state:
        render_prediction_summary(
            st.session_state["single_prediction"],
            st.session_state["single_request"],
            st.session_state["single_property"],
        )

with batch_tab:
    st.markdown("### Phân tích và ưu tiên danh mục")
    st.write(
        "Tải CSV theo schema `data/test.csv`. Dashboard sẽ trả dự báo, khoảng bất định, "
        "cảnh báo đầu vào và danh sách hồ sơ cần rà soát."
    )
    template = pd.DataFrame([{**NUMERIC_DEFAULTS, **{key: values[0] for key, values in CATEGORY_OPTIONS.items()}}])
    st.download_button(
        "Tải CSV mẫu", template.to_csv(index=False), "lendscope_input_template.csv", "text/csv"
    )
    uploaded = st.file_uploader("Chọn tệp CSV", type="csv", key="batch_upload")
    if uploaded is not None:
        input_frame = pd.read_csv(uploaded)
        st.caption(f"Đã đọc {len(input_frame):,} hồ sơ và {input_frame.shape[1]} cột.")
        st.dataframe(input_frame.head(15), use_container_width=True)
        if st.button("Chạy dự báo danh mục", type="primary", use_container_width=True):
            try:
                with st.spinner("Đang chấm điểm danh mục..."):
                    prediction = api_predict(json_records(input_frame))
                st.session_state["batch_output"] = pd.concat(
                    [input_frame.reset_index(drop=True), prediction], axis=1
                )
            except requests.RequestException as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                st.error(f"Dự báo thất bại: {detail}")

    if "batch_output" in st.session_state:
        output = st.session_state["batch_output"]
        prediction = output[
            ["predicted_sanction_amount_usd", "manual_review", "input_warning"]
        ]
        one, two, three, four = st.columns(4)
        one.metric("Tổng hồ sơ", f"{len(output):,}")
        two.metric("Hạn mức trung vị", money(float(prediction["predicted_sanction_amount_usd"].median())))
        three.metric("Cần rà soát", f"{int(prediction['manual_review'].sum()):,}")
        four.metric("Có cảnh báo đầu vào", f"{int(prediction['input_warning'].astype(bool).sum()):,}")

        chart_data = prediction[["predicted_sanction_amount_usd"]].rename(
            columns={"predicted_sanction_amount_usd": "Hạn mức ước lượng"}
        )
        st.markdown("#### Phân phối hạn mức ước lượng")
        st.bar_chart(chart_data["Hạn mức ước lượng"].value_counts(bins=12).sort_index())

        review_only = st.toggle("Chỉ hiển thị hồ sơ cần rà soát")
        displayed = output[output["manual_review"]] if review_only else output
        st.dataframe(displayed, use_container_width=True, hide_index=True)
        buffer = io.StringIO()
        output.to_csv(buffer, index=False)
        st.download_button(
            "Tải kết quả đầy đủ", buffer.getvalue(), "lendscope_predictions.csv", "text/csv"
        )

with governance_tab:
    st.markdown("### Hiệu năng đã khóa trên holdout")
    metrics = model.get("metrics", {})
    metric_rows = {
        "MAE": money(float(metrics.get("mae", 0))),
        "RMSE": money(float(metrics.get("rmse", 0))),
        "R²": f"{float(metrics.get('r2', 0)):.4f}",
        "WAPE": f"{float(metrics.get('wape', 0)):.2%}",
        "Coverage khoảng dự báo": f"{float(metrics.get('prediction_interval_90_coverage', 0)):.2%}",
        "Tỷ lệ rà soát thủ công": f"{float(metrics.get('manual_review_rate', 0)):.2%}",
    }
    st.dataframe(pd.DataFrame(metric_rows.items(), columns=["Chỉ số", "Giá trị"]), hide_index=True)

    st.markdown("### Luồng quyết định")
    st.code(
        "Tiền thẩm định → Ước lượng hạn mức → Áp trần nghiệp vụ → Đánh giá bất định → Phê duyệt chuyên viên",
        language=None,
    )
    st.markdown("### Các kiểm soát đang có")
    st.markdown(
        """
        - Không sử dụng ID và giới tính làm feature sản xuất.
        - Khoảng dự báo được hiệu chỉnh từ validation, không dùng holdout.
        - Dự báo không vượt số tiền yêu cầu và 80% giá trị tài sản.
        - Đầu vào ngoài p01–p99 hoặc category mới được cảnh báo.
        - `Property Age` trong dữ liệu nguồn gần như trùng hoàn toàn với `Income`; đây là vấn đề chất lượng dữ liệu cần xác minh với data owner trước production thật.
        """
    )
    st.markdown("### Điều mô hình không trả lời")
    st.info(
        "Mô hình không dự đoán hồ sơ có được duyệt hay không, không đo PD/LGD/ECL và không chứng minh hạn mức an toàn về rủi ro tín dụng."
    )

st.divider()
st.caption(
    "LendScope là hệ thống hỗ trợ quyết định dựa trên chính sách lịch sử. Mọi đề xuất cần nằm trong quy trình kiểm soát tín dụng và giám sát mô hình."
)

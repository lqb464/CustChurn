"""Định nghĩa schema dùng thống nhất cho bài toán hồi quy hạn mức vay."""

TARGET_COLUMN = "Loan Sanction Amount (USD)"
REQUEST_COLUMN = "Loan Amount Request (USD)"
SENTINEL_VALUE = -999
RANDOM_STATE = 42

ID_COLUMNS = ["Customer ID", "Name", "Property ID"]
SENSITIVE_COLUMNS = ["Gender"]

RAW_FEATURES = [
    "Age",
    "Income (USD)",
    "Income Stability",
    "Profession",
    "Type of Employment",
    "Location",
    REQUEST_COLUMN,
    "Current Loan Expenses (USD)",
    "Expense Type 1",
    "Expense Type 2",
    "Dependents",
    "Credit Score",
    "No. of Defaults",
    "Has Active Credit Card",
    "Property Age",
    "Property Type",
    "Property Location",
    "Co-Applicant",
    "Property Price",
]

ENGINEERED_FEATURES = [
    "RequestToIncome",
    "RequestToProperty",
    "ExpenseToIncome",
    "IncomePerDependent",
    "PropertyEquityProxy",
    "CreditIncomeInteraction",
    "CreditRequestInteraction",
    "AgeIncomeInteraction",
]

MODEL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES

NUMERIC_FEATURES = [
    "Age",
    "Income (USD)",
    REQUEST_COLUMN,
    "Current Loan Expenses (USD)",
    "Dependents",
    "Credit Score",
    "No. of Defaults",
    "Property Age",
    "Property Type",
    "Co-Applicant",
    "Property Price",
    *ENGINEERED_FEATURES,
]

CATEGORICAL_FEATURES = [
    "Income Stability",
    "Profession",
    "Type of Employment",
    "Location",
    "Expense Type 1",
    "Expense Type 2",
    "Has Active Credit Card",
    "Property Location",
]

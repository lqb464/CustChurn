import numpy as np
import pandas as pd
import pytest

from src.data.schema import REQUEST_COLUMN, TARGET_COLUMN


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = 120
    request = rng.uniform(20_000, 200_000, rows)
    return pd.DataFrame(
        {
            "Customer ID": [f"C{i}" for i in range(rows)], "Name": [f"N{i}" for i in range(rows)],
            "Property ID": [f"P{i}" for i in range(rows)], "Gender": rng.choice(["M", "F"], rows),
            "Age": rng.integers(21, 65, rows), "Income (USD)": rng.uniform(1_000, 10_000, rows),
            "Income Stability": rng.choice(["Low", "High"], rows),
            "Profession": rng.choice(["Working", "Commercial associate"], rows),
            "Type of Employment": rng.choice(["Managers", "Sales staff"], rows),
            "Location": rng.choice(["Urban", "Semi-Urban"], rows), REQUEST_COLUMN: request,
            "Current Loan Expenses (USD)": rng.uniform(50, 1_500, rows),
            "Expense Type 1": rng.choice(["Y", "N"], rows), "Expense Type 2": rng.choice(["Y", "N"], rows),
            "Dependents": rng.integers(0, 5, rows).astype(float), "Credit Score": rng.uniform(600, 850, rows),
            "No. of Defaults": rng.integers(0, 2, rows),
            "Has Active Credit Card": rng.choice(["Active", "Inactive"], rows),
            "Property Age": rng.uniform(5, 40, rows), "Property Type": rng.integers(1, 5, rows),
            "Property Location": rng.choice(["Urban", "Rural"], rows), "Co-Applicant": rng.integers(0, 2, rows),
            "Property Price": request * rng.uniform(1.1, 2.0, rows),
            TARGET_COLUMN: request * rng.choice([0.65, 0.70, 0.75, 0.80], rows),
        }
    )

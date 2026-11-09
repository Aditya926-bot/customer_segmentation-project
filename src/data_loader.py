
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "Invoice ID",
    "Customer ID",
    "Product ID",
    "Unit Price",
    "Quantity",
    "Date",
    "Discount",
    "Line Total",
    "Transaction Type",
]


def load_transactions(file_path):
    """Load and validate a CSV, Excel, or Parquet transaction file."""

    path = Path(file_path)
    extension = path.suffix.lower()

    try:
        if extension == ".csv":
            df = pd.read_csv(path)

        elif extension in (".xlsx", ".xls"):
            df = pd.read_excel(path)

        elif extension == ".parquet":
            df = pd.read_parquet(path)

        else:
            raise ValueError(
                "Unsupported file format. Upload CSV, XLSX, or Parquet."
            )

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Could not read the uploaded file: {exc}"
        ) from exc

    if df.empty:
        raise ValueError("The uploaded file contains no transaction rows.")

    # Remove accidental whitespace around column names.
    df.columns = df.columns.astype(str).str.strip()

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    if df["Customer ID"].isna().any():
        raise ValueError(
            "Customer ID contains missing values. Please correct the file."
        )

    customer_count = df["Customer ID"].nunique()

    if customer_count != 1:
        raise ValueError(
            f"The uploaded file contains {customer_count} customers. "
            "Please upload transactions for exactly one customer."
        )

    return df


import numpy as np
import pandas as pd


ANALYSIS_DATE = pd.Timestamp("2026-01-01")


def build_customer_features(
    transactions,
    analysis_date=ANALYSIS_DATE
):
    """
    Convert one customer's transaction rows into one customer profile.

    The calculations follow the uploaded feature-engineering notebook.
    Returns a DataFrame containing one row.
    """

    df = transactions.copy()

    # Normalize column names and parse transaction dates.
    df.columns = df.columns.astype(str).str.strip()

    required = [
        "Invoice ID", "Customer ID", "Product ID",
        "Unit Price", "Quantity", "Date", "Discount",
        "Line Total", "Transaction Type"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    if df.empty:
        raise ValueError("No transaction rows were provided.")

    if df["Customer ID"].isna().any():
        raise ValueError("Customer ID contains missing values.")

    if df["Customer ID"].nunique() != 1:
        raise ValueError(
            "Please upload transactions for exactly one customer."
        )

    # Match the duplicate removal used in the notebook.
    if "Line" in df.columns:
        df = df.drop_duplicates(
            subset=["Invoice ID", "Line"],
            keep="first"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if df["Date"].isna().any():
        raise ValueError(
            "Some transaction dates are missing or invalid."
        )

    for col in ["Unit Price", "Quantity", "Discount", "Line Total"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[["Unit Price", "Quantity", "Discount", "Line Total"]].isna().any().any():
        raise ValueError(
            "Price, quantity, discount, or line total contains invalid values."
        )

    df["Transaction Type"] = (
        df["Transaction Type"].astype(str).str.strip()
    )

    sales = df[df["Transaction Type"] == "Sale"].copy()
    returns = df[df["Transaction Type"] == "Return"].copy()

    if sales.empty:
        raise ValueError(
            "This customer has no Sale transactions to build a profile."
        )

    analysis_date = pd.Timestamp(analysis_date)

    # Base customer identity.
    customer_id = df["Customer ID"].iloc[0]
    profile = {"Customer ID": customer_id}

    # Recency and purchase dates: sales only.
    last_purchase = sales["Date"].max()
    first_purchase = sales["Date"].min()

    profile["Last_Purchase_Date"] = last_purchase
    profile["Recency"] = (analysis_date - last_purchase).days

    profile["first_Purchase_Date"] = first_purchase
    profile["Customer_Lifetime"] = (
        last_purchase - first_purchase
    ).days

    # Frequency in the notebook counts distinct invoices in all rows.
    profile["Frequency"] = df["Invoice ID"].nunique()

    # Unique products in all transaction rows.
    profile["Unique_Products"] = df["Product ID"].nunique()

    # Monetary features.
    profile["Net_Monetary"] = df["Line Total"].sum()
    profile["Gross_Monetary"] = sales["Line Total"].sum()
    profile["Total_Quantity"] = sales["Quantity"].sum()
    profile["Average_Unit_Price"] = sales["Unit Price"].mean()
    profile["average_discount_rate"] = sales["Discount"].mean()

    # Average purchase interval.
    frequency_minus_one = profile["Frequency"] - 1
    if frequency_minus_one > 0:
        profile["Average_Purchase_Interval"] = (
            profile["Customer_Lifetime"] / frequency_minus_one
        )
    else:
        profile["Average_Purchase_Interval"] = 0.0

    # Quantities bought at base price or with a discount.
    profile["products_with_base_price"] = sales.loc[
        sales["Discount"] == 0, "Quantity"
    ].sum()

    profile["products_with_discount"] = sales.loc[
        sales["Discount"] > 0, "Quantity"
    ].sum()

    # Return-related features.
    profile["Return_Count"] = returns["Invoice ID"].nunique()
    profile["Returned_Quantity"] = returns["Quantity"].sum()
    profile["Return_Value"] = returns["Line Total"].sum().__abs__()

    # Most common shopping day and month.
    profile["Favorite_Shopping_Day_in_week"] = (
        sales["Date"].dt.day_name().mode().iloc[0]
    )
    profile["Favorite_Shopping_month_in_year"] = (
        sales["Date"].dt.month_name().mode().iloc[0]
    )

    # Monthly spending statistics: sales only.
    monthly = (
        sales.assign(YearMonth=sales["Date"].dt.to_period("M"))
        .groupby("YearMonth")["Line Total"]
        .sum()
        .sort_index()
    )

    profile["Monthly_Spending_Mean"] = monthly.mean()
    profile["Monthly_Spending_Std"] = monthly.std()

    if len(monthly) < 2:
        profile["Spending_Trend"] = 0.0
    else:
        x = np.arange(len(monthly))
        profile["Spending_Trend"] = float(
            np.polyfit(x, monthly.to_numpy(), 1)[0]
        )

    # Derived features.
    profile["Is_One_Time_Customer"] = int(
        profile["Frequency"] == 1
    )

    sale_frequency = sales["Invoice ID"].nunique()
    profile["Sale_Frequency"] = sale_frequency

    profile["average_order_value"] = (
        profile["Gross_Monetary"] / sale_frequency
        if sale_frequency else 0.0
    )

    profile["Return_Rate"] = (
        profile["Return_Count"] / sale_frequency
        if sale_frequency else 0.0
    )

    # Match the notebook's missing-value handling.
    for col in [
        "products_with_base_price",
        "products_with_discount",
        "Return_Count",
        "Returned_Quantity",
        "Return_Value",
        "Average_Purchase_Interval",
        "Monthly_Spending_Std",
    ]:
        if pd.isna(profile[col]):
            profile[col] = 0.0

    if profile["Net_Monetary"] < 0:
        raise ValueError(
            "This customer's Net_Monetary is negative, "
            "so the customer would have been excluded from model training."
        )

    return pd.DataFrame([profile])

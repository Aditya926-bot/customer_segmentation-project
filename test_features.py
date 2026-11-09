
import pandas as pd

from src.features import build_customer_features
from src.data_loader import load_transactions

transactions = load_transactions(
    "sample_data/sample_customer_2025.csv"
)

customer_features = build_customer_features(transactions)

print("Customer feature shape:", customer_features.shape)
print("\nFeatures used by the KMeans model:")

model_features = [
    "Recency",
    "Sale_Frequency",
    "Monthly_Spending_Mean",
    "average_discount_rate",
    "Average_Unit_Price",
]

print(customer_features[model_features].to_string(index=False))

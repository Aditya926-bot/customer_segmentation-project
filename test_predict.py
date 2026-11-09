
from src.data_loader import load_transactions
from src.features import build_customer_features
from src.predict import predict_customer


file_path = "sample_data/sample_customer_2025.csv"

transactions = load_transactions(file_path)
customer_features = build_customer_features(transactions)

result = predict_customer(customer_features)

print("\nCUSTOMER SEGMENTATION RESULT")
print("----------------------------")
print("Customer ID:", result["customer_id"])
print("Cluster:", result["cluster_id"])
print("Segment:", result["segment_name"])

print("\nDistance to assigned cluster center:",
      round(result["distance_to_cluster_center"], 4))

print("\nDistance to runner-up cluster:",
      round(result["runner_up_distance"], 4)
      if result["runner_up_distance"] is not None else "N/A")

print("\nFEATURE DISTANCE EXPLANATION")
for reason in result["feature_reasons"]:
    print(
        reason["feature"],
        "difference:",
        round(reason["difference"], 4)
    )

print("\nCUSTOMER PROFILE COMPARISON")
for reason in result["profile_reasons"]:
    print(
        reason["feature"],
        "customer:", round(reason["customer_value"], 2),
        "cluster median:", round(reason["cluster_median"], 2),
        "comparison:", reason["comparison"]
    )

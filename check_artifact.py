
import pickle
import sklearn
from pathlib import Path

artifact_path = Path("artifacts/artifacts.pkl")

if not artifact_path.exists():
    raise FileNotFoundError(
        "Place your updated artifacts.pkl in the artifacts folder."
    )

with artifact_path.open("rb") as f:
    artifacts = pickle.load(f)

print("Artifact loaded successfully!")
print("Installed scikit-learn:", sklearn.__version__)
print("Model features:", artifacts["features"])
print("Segment names:")

for cluster_id, name in artifacts["segment_names"].items():
    print(f"Cluster {cluster_id}: {name}")

print("Saved scikit-learn version:",
      artifacts.get("sklearn_version"))

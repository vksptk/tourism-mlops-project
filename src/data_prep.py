import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

TARGET_COL = "ProdTaken"

def main():
    # Read repo ids from env vars or use defaults
    hf_dataset_id = os.getenv("HF_DATASET_REPO", "vksptk/tourism-dataset")
    project_name = os.getenv("PROJECT_NAME", "tourism_mlops_project")

    print(f"Using dataset repo: {hf_dataset_id}")
    print(f"Project name: {project_name}")

    # Load dataset from Hugging Face
    raw_dataset = load_dataset(hf_dataset_id)
    df = raw_dataset["train"].to_pandas()
    print("Raw shape:", df.shape)

    # ----- Cleaning -----
    data = df.copy()

    # Drop index column if present
    if "Unnamed: 0" in data.columns:
        data.drop(columns=["Unnamed: 0"], inplace=True)
        print("Dropped column 'Unnamed: 0'.")

    # Drop duplicates
    before = data.shape[0]
    data.drop_duplicates(inplace=True)
    after = data.shape[0]
    print(f"Removed {before - after} duplicate rows.")

    print("Columns:", list(data.columns))
    print("Missing values per column:\n", data.isna().sum())
    print("Cleaned shape:", data.shape)

    # ----- Train/Test split -----
    X = data.drop(columns=[TARGET_COL])
    y = data[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train

    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test

    # Local save paths
    data_dir = os.path.join(project_name, "data")
    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Train saved to:", train_path)
    print("Test saved to:", test_path)

    # ----- Upload to HF dataset repo -----
    api = HfApi()

    api.upload_file(
        path_or_fileobj=train_path,
        path_in_repo="train.csv",
        repo_id=hf_dataset_id,
        repo_type="dataset"
    )
    api.upload_file(
        path_or_fileobj=test_path,
        path_in_repo="test.csv",
        repo_id=hf_dataset_id,
        repo_type="dataset"
    )

    print("Uploaded train.csv and test.csv back to HF dataset:", hf_dataset_id)

if __name__ == "__main__":
    main()

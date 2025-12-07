import os
import pandas as pd
import joblib

from huggingface_hub import HfApi, hf_hub_download

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

TARGET_COL = "ProdTaken"

def main():
    project_name = os.getenv("PROJECT_NAME", "tourism_mlops_project")
    hf_dataset_id = os.getenv("HF_DATASET_REPO", "vksptk/tourism-dataset")
    hf_model_repo = os.getenv("HF_MODEL_REPO", "vksptk/tourism-wellness-package-model")

    print(f"Using dataset repo: {hf_dataset_id}")
    print(f"Using model repo:   {hf_model_repo}")

    # ----- Download train & test from HF dataset -----
    train_local_path = hf_hub_download(
        repo_id=hf_dataset_id,
        repo_type="dataset",
        filename="train.csv"
    )
    test_local_path = hf_hub_download(
        repo_id=hf_dataset_id,
        repo_type="dataset",
        filename="test.csv"
    )

    train_df = pd.read_csv(train_local_path)
    test_df = pd.read_csv(test_local_path)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    # Identify columns
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(exclude=["object"]).columns.tolist()

    print("Categorical cols:", categorical_cols)
    print("Numeric cols:", numeric_cols)

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    # Simple RF model (can be tuned more if needed)
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", rf_model)
    ])

    # ----- Train -----
    pipeline.fit(X_train, y_train)
    print("Model training completed.")

    # ----- Evaluate -----
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # ----- Save model locally -----
    model_dir = os.path.join(project_name, "model")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "best_model.joblib")
    joblib.dump(pipeline, model_path)
    print("Saved model to:", model_path)

    # ----- Upload model to HF model hub -----
    api = HfApi()
    api.create_repo(
        repo_id=hf_model_repo,
        repo_type="model",
        private=False,
        exist_ok=True
    )

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_model.joblib",
        repo_id=hf_model_repo,
        repo_type="model"
    )

    print("Uploaded model to HF model hub:", hf_model_repo)

if __name__ == "__main__":
    main()

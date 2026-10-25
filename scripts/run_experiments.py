import argparse
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot_detector.config import Config, load_config, set_seed
from src.bot_detector.cv import get_splits
from src.bot_detector.data import load_split, participant_texts
from src.bot_detector.features import NUMERIC_FEATURE_COLS, build_features
from src.bot_detector.metrics import compute_classification_metrics
from src.bot_detector.models import BaseModel, CnnMlpModel, LGBMTfidfModel, TfidfLogRegModel

REQUIRED_DATA_FILES = ["train.json", "test.json", "ytrain.csv", "ytest.csv"]


def check_data(cfg: Config) -> None:
    data_dir = cfg.path("data_dir")
    missing = [f for f in REQUIRED_DATA_FILES if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError()


def resolve_tracking_uri(cfg: Config) -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI") or cfg["mlflow"]["tracking_uri"]
    if uri.startswith(("http://", "https://", "file:", "sqlite://")):
        return uri
    p = Path(uri)
    if not p.is_absolute():
        p = cfg.project_root / p
    p.mkdir(parents=True, exist_ok=True)
    return p.as_uri()


def build_model(name: str, cfg: Config) -> BaseModel:
    params = cfg["models"].get(name, {})
    if name == "cnn_mlp":
        return CnnMlpModel(random_state=cfg["seed"], **params)
    vec = dict(params.get("vectorizer", {}))
    clf = dict(params.get("clf", {}))
    if name == "tfidf_logreg":
        return TfidfLogRegModel(random_state=cfg["seed"], **vec, **clf)
    if name == "lgbm_tfidf":
        return LGBMTfidfModel(random_state=cfg["seed"], **vec, **clf)
    raise ValueError(f"unknown model: {name}")


def run_pipeline(cfg: Config) -> dict:
    set_seed(cfg["seed"])
    runs_dir = cfg.path("runs_dir")
    runs_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(resolve_tracking_uri(cfg))
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    split = load_split(cfg.path("data_dir"))
    train_msgs, test_msgs = split["train_msgs"], split["test_msgs"]

    train_texts = participant_texts(train_msgs)
    test_texts = participant_texts(test_msgs)
    train_feats = build_features(train_msgs)
    test_feats = build_features(test_msgs)

    train_df = train_texts.merge(
        train_feats, on=["dialog_id", "participant_index"], how="inner"
    )
    test_df = test_texts.merge(
        test_feats, on=["dialog_id", "participant_index"], how="inner"
    )

    labels = split["train_labels"]
    train_df = train_df.merge(
        labels, on=["dialog_id", "participant_index"], how="inner"
    )

    y = train_df["is_bot"].astype(int).to_numpy()
    groups = train_df["dialog_id"].to_numpy()
    n_splits = cfg["cv"]["n_splits"]
    folds = get_splits(y, groups, n_splits=n_splits, seed=cfg["seed"])

    feature_cols = ["text", *NUMERIC_FEATURE_COLS]
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    print(f"train participants: {len(train_df)} | test: {len(test_df)} | folds: {n_splits}")
    print(f"bot share: overall={y.mean():.3f}")

    results: dict[str, dict] = {}
    best_name, best_cv_logloss, best_test_pred = None, np.inf, None

    for name in cfg["experiments"]:
        print(f"\n=== {name} ===")
        fold_metrics = []
        for fold, (tr_idx, va_idx) in enumerate(folds):
            model = build_model(name, cfg)
            model.fit(
                X_train.iloc[tr_idx], y[tr_idx],
                X_train.iloc[va_idx], y[va_idx],
            )
            pred = model.predict_proba(X_train.iloc[va_idx])[:, 1]
            m = compute_classification_metrics(y[va_idx], pred)
            fold_metrics.append(m)
            print(f"  fold {fold}: " + " | ".join(f"{k}={v:.4f}" for k, v in m.items()))

        metric_names = list(fold_metrics[0])
        cv_metrics = {
            metric: float(np.mean([m[metric] for m in fold_metrics]))
            for metric in metric_names
        }
        cv_metrics.update(
            {
                f"{metric}_std": float(np.std([m[metric] for m in fold_metrics]))
                for metric in metric_names
            }
        )
        cv_logloss = cv_metrics["logloss"]
        print("CV mean: " + " | ".join(f"{k}={v:.4f}" for k, v in cv_metrics.items()))

        final_model = build_model(name, cfg)
        va_idx_last = folds[-1][1]
        final_model.fit(X_train, y, X_train.iloc[va_idx_last], y[va_idx_last])
        test_pred = final_model.predict_proba(X_test)[:, 1]

        with mlflow.start_run(run_name=name):
            mlflow.set_tag("experiment_type", name)
            mlflow.log_params(
                {
                    **{k: str(v) for k, v in model.get_params().items()},
                    "seed": cfg["seed"],
                    "n_splits": n_splits,
                    "n_numeric_features": len(NUMERIC_FEATURE_COLS),
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                }
            )
            per_fold = {}
            for fold, m in enumerate(fold_metrics):
                per_fold.update({f"{k}_fold_{fold}": v for k, v in m.items()})
            mlflow.log_metrics(per_fold)
            mlflow.log_metrics(cv_metrics)

            test_path = runs_dir / f"test_{name}.parquet"
            pd.DataFrame(
                {
                    "dialog_id": test_df["dialog_id"],
                    "participant_index": test_df["participant_index"],
                    "y_pred": test_pred,
                }
            ).to_parquet(test_path, index=False)
            mlflow.log_artifact(str(test_path))

            if hasattr(final_model, "feature_importances"):
                fi = final_model.feature_importances()
                fi_path = runs_dir / f"feature_importance_{name}.csv"
                fi.to_csv(fi_path, index=False)
                mlflow.log_artifact(str(fi_path))

        results[name] = cv_metrics
        if cv_logloss < best_cv_logloss:
            best_cv_logloss = cv_logloss
            best_name = name
            best_test_pred = test_pred

    pred_df = pd.DataFrame(
        {
            "dialog_id": test_df["dialog_id"],
            "participant_index": test_df["participant_index"],
            "y_pred": best_test_pred,
        }
    )
    test_ids = split["test_ids"]
    sub = test_ids[["ID", "dialog_id", "participant_index"]].merge(
        pred_df, on=["dialog_id", "participant_index"], how="left"
    )
    sub = sub[["ID", "y_pred"]].rename(columns={"y_pred": "is_bot"})

    sub_path = runs_dir / "submission.csv"
    sub.to_csv(sub_path, index=False)

    print(f"\nBest model: {best_name} | CV logloss = {best_cv_logloss:.4f}")
    print(f"Submission saved to {sub_path}")

    results["_best"] = {
        "name": best_name,
        "cv_logloss": best_cv_logloss,
        "submission_path": str(sub_path),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    check_data(cfg)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
import json
from pathlib import Path

import pandas as pd


def load_dialogs(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_labels(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def dialog_to_df(dialogs: dict) -> pd.DataFrame:
    rows = []
    for dialog_id, messages in dialogs.items():
        for m in messages:
            rows.append(
                {
                    "dialog_id": dialog_id,
                    "participant_index": int(m["participant_index"]),
                    "message": int(m["message"]),
                    "text": m.get("text", "") or "",
                }
            )
    return pd.DataFrame(rows)


def load_split(data_dir: Path) -> dict:
    train_msgs = dialog_to_df(load_dialogs(data_dir / "train.json"))
    test_msgs = dialog_to_df(load_dialogs(data_dir / "test.json"))
    train_labels = load_labels(data_dir / "ytrain.csv")
    test_ids = load_labels(data_dir / "ytest.csv")
    return {
        "train_msgs": train_msgs,
        "test_msgs": test_msgs,
        "train_labels": train_labels,
        "test_ids": test_ids,
    }


def participant_texts(msgs: pd.DataFrame) -> pd.DataFrame:
    ordered = msgs.sort_values(["dialog_id", "participant_index", "message"])
    grouped = (
        ordered.groupby(["dialog_id", "participant_index"], sort=False)["text"]
        .apply(lambda s: "\n".join(s.tolist()))
        .reset_index()
    )
    grouped.columns = ["dialog_id", "participant_index", "text"]
    return grouped


def merge_features_labels(feats: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    assert len(feats) == len(labels), (
        f"row mismatch: features={len(feats)} labels={len(labels)}"
    )
    key = ["dialog_id", "participant_index"]
    assert not feats.duplicated(key).any(), "duplicate keys in features"
    assert not labels.duplicated(key).any(), "duplicate keys in labels"
    merged = feats.merge(labels, on=key, how="left")
    assert merged["is_bot"].notna().all(), "NaN labels after merge"
    return merged
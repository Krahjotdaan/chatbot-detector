import re
import numpy as np
import pandas as pd

WORD_RE = re.compile(r"\w+", re.UNICODE)
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def participant_internal_features(group: pd.DataFrame) -> dict:
    lengths = group["text"].str.len().to_numpy()
    words = group["text"].str.split().apply(len).to_numpy()
    texts = group["text"].tolist()

    all_tokens = [t for txt in texts for t in tokenize(txt)]
    unique_tokens = set(all_tokens)

    ends_punct = np.mean([t.strip().endswith((".", "!", "?")) for t in texts]) if texts else 0.0
    starts_upper = np.mean([t.strip()[:1].isupper() for t in texts]) if texts else 0.0
    has_digits = np.mean([any(ch.isdigit() for ch in t) for t in texts]) if texts else 0.0
    has_latin = np.mean([bool(re.search(r"[a-zA-Z]", t)) for t in texts]) if texts else 0.0
    has_emoji = np.mean([bool(EMOJI_RE.search(t)) for t in texts]) if texts else 0.0
    has_markdown = np.mean(
        [bool(re.search(r"(\*\*|__|^#{1,3}\s|`{1,3}|^\s*[-*]\s)", t, re.M)) for t in texts]
    ) if texts else 0.0
    has_multiline = np.mean(["\n\n" in t for t in texts]) if texts else 0.0

    return {
        "n_messages": len(texts),
        "mean_len_chars": float(np.mean(lengths)) if len(lengths) else 0.0,
        "median_len_chars": float(np.median(lengths)) if len(lengths) else 0.0,
        "std_len_chars": float(np.std(lengths)) if len(lengths) else 0.0,
        "min_len_chars": int(np.min(lengths)) if len(lengths) else 0,
        "max_len_chars": int(np.max(lengths)) if len(lengths) else 0,
        "mean_len_words": float(np.mean(words)) if len(words) else 0.0,
        "ends_with_punct_rate": float(ends_punct),
        "starts_uppercase_rate": float(starts_upper),
        "digits_rate": float(has_digits),
        "latin_rate": float(has_latin),
        "emoji_rate": float(has_emoji),
        "markdown_rate": float(has_markdown),
        "multiline_rate": float(has_multiline),
        "unique_words_ratio": len(unique_tokens) / max(len(all_tokens), 1),
    }


def participant_reactive_features(group: pd.DataFrame, all_msgs: pd.DataFrame) -> dict:
    pidx = group["participant_index"].iloc[0]
    dialog_id = group["dialog_id"].iloc[0]
    dialog_msgs = (
        all_msgs[all_msgs["dialog_id"] == dialog_id]
        .sort_values("message")
        .reset_index(drop=True)
    )

    echo_hits = 0
    self_repeat_hits = 0
    lexical_overlaps: list[float] = []
    answer_to_question: list[bool] = []
    gaps: list[int] = []
    consecutive = 0

    prev_participant_msg_num = None
    prev_own_text = None

    for i, row in dialog_msgs.iterrows():
        if row["participant_index"] != pidx:
            continue

        prev_other_idx = None
        for j in range(i - 1, -1, -1):
            if dialog_msgs.loc[j, "participant_index"] != pidx:
                prev_other_idx = j
                break

        if prev_other_idx is not None:
            gap = int(row["message"] - dialog_msgs.loc[prev_other_idx, "message"])
            gaps.append(gap)
            other_text = dialog_msgs.loc[prev_other_idx, "text"]
            if other_text.strip().lower() == row["text"].strip().lower():
                echo_hits += 1
            lexical_overlaps.append(
                jaccard(set(tokenize(other_text)), set(tokenize(row["text"])))
            )
            if other_text.strip().endswith("?"):
                answer_to_question.append(len(row["text"].strip()) > 0)

        if prev_own_text is not None and prev_own_text.strip().lower() == row["text"].strip().lower():
            self_repeat_hits += 1
        prev_own_text = row["text"]

        if prev_participant_msg_num is not None and row["message"] - prev_participant_msg_num == 1:
            consecutive += 1
        prev_participant_msg_num = row["message"]

    n = len(group)
    return {
        "echo_rate": echo_hits / n if n else 0.0,
        "self_repeat_rate": self_repeat_hits / n if n else 0.0,
        "mean_lexical_overlap": float(np.mean(lexical_overlaps)) if lexical_overlaps else 0.0,
        "answer_to_question_rate": float(np.mean(answer_to_question)) if answer_to_question else 0.0,
        "mean_gap": float(np.mean(gaps)) if gaps else 0.0,
        "max_gap": int(np.max(gaps)) if gaps else 0,
        "n_consecutive": consecutive,
    }


def participant_dialog_features(group: pd.DataFrame, dialog_len: int) -> dict:
    first_message_overall = int(group["message"].min())
    return {
        "is_first_speaker": int(first_message_overall == 0),
        "first_msg_index": first_message_overall,
        "share_of_dialog": len(group) / max(dialog_len, 1),
        "dialog_len": dialog_len,
    }


def add_pair_features(feats: pd.DataFrame) -> pd.DataFrame:
    feats = feats.copy()
    for col in [
        "n_messages", "mean_len_chars", "echo_rate",
        "mean_lexical_overlap", "n_consecutive",
    ]:
        other = (
            feats.groupby("dialog_id")[col]
            .transform("sum")
            .sub(feats[col])
        )
        feats[f"{col}_other"] = other
        feats[f"{col}_ratio"] = feats[col] / (other + 1e-6)
    return feats


def build_features(dialogs_df: pd.DataFrame) -> pd.DataFrame:
    dialog_lens = dialogs_df.groupby("dialog_id").size().to_dict()
    rows = []
    for (dialog_id, pidx), group in dialogs_df.groupby(["dialog_id", "participant_index"]):
        group = group.sort_values("message").reset_index(drop=True)
        row = {
            "dialog_id": dialog_id,
            "participant_index": pidx,
            **participant_internal_features(group),
            **participant_reactive_features(group, dialogs_df),
            **participant_dialog_features(group, dialog_lens[dialog_id]),
        }
        rows.append(row)
    feats = pd.DataFrame(rows)
    feats = add_pair_features(feats)
    return feats


def participant_text(dialogs_df: pd.DataFrame, dialog_id: str, pidx: int) -> str:
    """Concatenated text of one participant — used by TF-IDF model."""
    sub = dialogs_df[
        (dialogs_df["dialog_id"] == dialog_id) & (dialogs_df["participant_index"] == pidx)
    ]
    return "\n".join(sub.sort_values("message")["text"].tolist())


NUMERIC_FEATURE_COLS = [
    "n_messages", "mean_len_chars", "median_len_chars", "std_len_chars",
    "min_len_chars", "max_len_chars", "mean_len_words",
    "ends_with_punct_rate", "starts_uppercase_rate",
    "digits_rate", "latin_rate", "emoji_rate", "markdown_rate", "multiline_rate",
    "unique_words_ratio",
    "echo_rate", "self_repeat_rate", "mean_lexical_overlap",
    "answer_to_question_rate", "mean_gap", "max_gap", "n_consecutive",
    "is_first_speaker", "first_msg_index", "share_of_dialog", "dialog_len",
    "n_messages_other", "n_messages_ratio",
    "mean_len_chars_other", "mean_len_chars_ratio",
    "echo_rate_other", "echo_rate_ratio",
    "mean_lexical_overlap_other", "mean_lexical_overlap_ratio",
    "n_consecutive_other", "n_consecutive_ratio",
]
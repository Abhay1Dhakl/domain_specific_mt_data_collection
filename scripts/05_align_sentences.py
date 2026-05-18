import re
import pandas as pd
from pathlib import Path


SENTENCE_CANDIDATES_PATH = Path("data/sentences/sentence_candidates.csv")
ALIGNED_OUTPUT_PATH = Path("data/aligned/sentence_pairs_raw.csv")
MANUAL_REVIEW_OUTPUT_PATH = Path("data/aligned/manual_review_template.csv")
ALIGNMENT_LOG_PATH = Path("data/logs/alignment_log.csv")


def clean_for_alignment(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def is_too_short(text: str, lang: str) -> bool:
    text = str(text).strip()

    if lang == "en":
        return len(text.split()) < 2

    if lang == "ne":
        # Nepali word split by spaces
        return len(text.split()) < 2

    return len(text) < 3


def is_heading_like(text: str) -> bool:
    """
    Detects headings or very short labels.
    We do not remove them completely, but we mark them as low confidence.
    """
    text = str(text).strip()

    if len(text) <= 40 and not re.search(r"[।.!?]$", text):
        return True

    return False


def length_ratio_flag(en: str, ne: str) -> tuple[float, str]:
    """
    Rough length-ratio check.
    Nepali and English character lengths do not match perfectly,
    but extreme ratios usually indicate bad alignment.
    """
    en_len = max(len(en), 1)
    ne_len = max(len(ne), 1)

    ratio = ne_len / en_len

    if 0.35 <= ratio <= 3.0:
        return ratio, "ok"
    elif 0.20 <= ratio < 0.35 or 3.0 < ratio <= 4.5:
        return ratio, "warning"
    else:
        return ratio, "bad"


def assign_alignment_quality(en: str, ne: str, ratio_status: str) -> str:
    """
    Assigns a simple quality label for manual review.
    """
    if is_too_short(en, "en") or is_too_short(ne, "ne"):
        return "needs_review_short"

    if is_heading_like(en) or is_heading_like(ne):
        return "needs_review_heading"

    if ratio_status == "bad":
        return "needs_review_length_mismatch"

    if ratio_status == "warning":
        return "medium_confidence"

    return "high_confidence"


def main():
    df = pd.read_csv(SENTENCE_CANDIDATES_PATH)

    required_columns = [
        "global_sentence_id",
        "source_id",
        "title",
        "subdomain",
        "paragraph_id",
        "local_sentence_id",
        "language",
        "sentence",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in sentence_candidates.csv: {missing}")

    # Keep only English and Nepali sentence candidates
    df = df[df["language"].isin(["en", "ne"])].copy()

    # Sort by document order
    df = df.sort_values(
        by=["source_id", "paragraph_id", "local_sentence_id", "global_sentence_id"]
    )

    aligned_rows = []
    log_rows = []

    pair_counter = 1

    for source_id, group in df.groupby("source_id"):
        group = group.copy()

        title = group["title"].iloc[0]
        subdomain = group["subdomain"].iloc[0]

        en_df = group[group["language"] == "en"].copy()
        ne_df = group[group["language"] == "ne"].copy()

        en_df["sentence"] = en_df["sentence"].apply(clean_for_alignment)
        ne_df["sentence"] = ne_df["sentence"].apply(clean_for_alignment)

        en_df = en_df[en_df["sentence"].str.len() > 0]
        ne_df = ne_df[ne_df["sentence"].str.len() > 0]

        num_en = len(en_df)
        num_ne = len(ne_df)
        num_pairs = min(num_en, num_ne)

        print(f"\nAligning {source_id} - {title}")
        print(f"EN: {num_en} | NE: {num_ne} | Pairs: {num_pairs}")

        en_records = en_df.to_dict("records")
        ne_records = ne_df.to_dict("records")

        for i in range(num_pairs):
            en_row = en_records[i]
            ne_row = ne_records[i]

            en_text = en_row["sentence"]
            ne_text = ne_row["sentence"]

            ratio, ratio_status = length_ratio_flag(en_text, ne_text)
            quality = assign_alignment_quality(en_text, ne_text, ratio_status)

            aligned_rows.append({
                "pair_id": f"PAIR_{pair_counter:07d}",
                "source_id": source_id,
                "title": title,
                "subdomain": subdomain,

                "en_sentence_id": en_row["global_sentence_id"],
                "ne_sentence_id": ne_row["global_sentence_id"],

                "en_paragraph_id": en_row["paragraph_id"],
                "ne_paragraph_id": ne_row["paragraph_id"],

                "en": en_text,
                "ne": ne_text,

                "en_chars": len(en_text),
                "ne_chars": len(ne_text),
                "length_ratio_ne_en": round(ratio, 3),
                "length_ratio_status": ratio_status,

                "alignment_method": "sequential_order_baseline",
                "quality_label": quality,
                "review_status": "pending"
            })

            pair_counter += 1

        log_rows.append({
            "source_id": source_id,
            "title": title,
            "subdomain": subdomain,
            "num_en_sentences": num_en,
            "num_ne_sentences": num_ne,
            "num_aligned_pairs": num_pairs,
            "num_unmatched_en": max(num_en - num_ne, 0),
            "num_unmatched_ne": max(num_ne - num_en, 0),
            "alignment_status": "aligned_first_pass"
        })

    aligned_df = pd.DataFrame(aligned_rows)
    log_df = pd.DataFrame(log_rows)

    ALIGNED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALIGNMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    aligned_df.to_csv(ALIGNED_OUTPUT_PATH, index=False)
    log_df.to_csv(ALIGNMENT_LOG_PATH, index=False)

    # Create manual review file
    manual_review_df = aligned_df[
        [
            "pair_id",
            "source_id",
            "title",
            "subdomain",
            "en",
            "ne",
            "quality_label",
            "length_ratio_status",
            "review_status",
        ]
    ].copy()

    manual_review_df["corrected_en"] = ""
    manual_review_df["corrected_ne"] = ""
    manual_review_df["review_notes"] = ""

    manual_review_df.to_csv(MANUAL_REVIEW_OUTPUT_PATH, index=False)

    print("\nAlignment complete.")
    print(f"Raw aligned pairs saved to: {ALIGNED_OUTPUT_PATH}")
    print(f"Manual review template saved to: {MANUAL_REVIEW_OUTPUT_PATH}")
    print(f"Alignment log saved to: {ALIGNMENT_LOG_PATH}")

    if not aligned_df.empty:
        print("\nQuality label counts:")
        print(aligned_df["quality_label"].value_counts())


if __name__ == "__main__":
    main()
import csv
import re
from pathlib import Path


SENTENCE_CANDIDATES_PATH = Path("data/sentences/sentence_candidates.csv")
ALIGNED_OUTPUT_PATH = Path("data/aligned/sentence_pairs_raw.csv")
MANUAL_REVIEW_OUTPUT_PATH = Path("data/aligned/manual_review_template.csv")
ALIGNMENT_LOG_PATH = Path("data/logs/alignment_log.csv")
FIGURE_LABEL_PATTERNS = [
    "retinal blood vessels",
    "back of eye",
    "front of eye",
]

INPUT_FIELDNAMES = [
    "global_sentence_id",
    "source_id",
    "title",
    "subdomain",
    "paragraph_id",
    "local_sentence_id",
    "language",
    "sentence",
]

ALIGNED_FIELDNAMES = [
    "pair_id",
    "source_id",
    "title",
    "subdomain",
    "en_sentence_id",
    "ne_sentence_id",
    "en_paragraph_id",
    "ne_paragraph_id",
    "en",
    "ne",
    "en_chars",
    "ne_chars",
    "length_ratio_ne_en",
    "length_ratio_status",
    "alignment_method",
    "quality_label",
    "review_status",
]

MANUAL_REVIEW_FIELDNAMES = [
    "pair_id",
    "source_id",
    "title",
    "subdomain",
    "en",
    "ne",
    "quality_label",
    "length_ratio_status",
    "review_status",
    "corrected_en",
    "corrected_ne",
    "review_notes",
]

LOG_FIELDNAMES = [
    "source_id",
    "title",
    "subdomain",
    "num_en_sentences",
    "num_ne_sentences",
    "num_aligned_pairs",
    "num_unmatched_en",
    "num_unmatched_ne",
    "num_run_pairs",
    "alignment_status",
]


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def natural_sentence_id(value: str) -> int:
    match = re.search(r"(\d+)$", str(value))
    return int(match.group(1)) if match else 0


def count_devanagari(text: str) -> int:
    return len(re.findall(r"[\u0900-\u097F]", text))


def count_latin(text: str) -> int:
    return len(re.findall(r"[A-Za-z]", text))


def normalize_comparison_text(text: str) -> str:
    text = str(text)
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("ﬁ", "fi")
    text = text.replace("ﬂ", "fl")
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .:-")


def clean_for_alignment(text: str) -> str:
    text = str(text).replace("Ì", "")
    text = text.replace("�", "")
    text = text.replace("ﬁ", "fi")
    text = text.replace("ﬂ", "fl")
    text = re.sub(r"[\uE000-\uF8FF]", "", text)
    text = re.sub(r"([।.!?ः])(?=[^\s])", r"\1 ", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"•\s+", "• ", text)
    return text


def is_urlish_fragment(text: str) -> bool:
    lower = str(text).lower()
    if not any(token in lower for token in ["www.", ".gov", "cdc.", "immunize.", "vaccineinformation."]):
        return False

    return len(str(text).split()) <= 10


def strip_header_prefixes(text: str, title: str) -> str:
    text = clean_for_alignment(text)
    title_pattern = re.escape(clean_for_alignment(title))

    patterns = [
        rf"^\d+\s+{title_pattern}\.?\s+Nepali\.?\s*",
        rf"^{title_pattern}\.?\s+Nepali\.?\s*",
        r"^\d+\s+healthinfotranslations\.org\s*",
        r"^healthinfotranslations\.org\s*",
        r"^Nepali\.?\s*",
    ]

    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    return clean_for_alignment(text)


def is_too_short(text: str, lang: str) -> bool:
    text = clean_for_alignment(text)

    if lang in {"en", "ne"}:
        return len(text.replace("•", " ").split()) < 2

    return len(text) < 3


def is_heading_like(text: str) -> bool:
    text = clean_for_alignment(text)

    if text.startswith("•"):
        return False

    if len(text.split()) <= 8 and not re.search(r"[।.!?:ः]$", text):
        return True

    return False


def is_alignment_noise(text: str, lang: str, title: str) -> bool:
    text = strip_header_prefixes(text, title)
    lower = text.lower()
    word_count = len(text.replace("•", " ").split())
    title_lower = normalize_comparison_text(title)

    if not text:
        return True

    if re.fullmatch(r"\d+", text):
        return True

    if normalize_comparison_text(text) in {title_lower, "nepali"}:
        return True

    if "healthinfotranslations.org" in lower:
        return True

    if any(pattern in lower for pattern in FIGURE_LABEL_PATTERNS):
        return True

    if any(
        phrase in lower
        for phrase in [
            "©",
            "copyright",
            "unless otherwise stated",
            "the medical information found on this website",
            "you should always seek the advice of your doctor",
            "a result of your stopping medical treatment",
            "the ohio state university",
            "wexner medical center",
            "mount carmel health system",
            "ohiohealth",
            "nationwide children",
        ]
    ):
        return True

    if is_urlish_fragment(text):
        return True

    if word_count <= 12 and not text.startswith("•") and not re.search(r"[।.!?:ः]$", text):
        return True

    if lang == "en" and count_latin(text) < 3:
        return True

    if lang == "ne" and count_devanagari(text) < 3:
        return True

    return False


def length_ratio_flag(en: str, ne: str) -> tuple[float, str]:
    en_len = max(len(en), 1)
    ne_len = max(len(ne), 1)
    ratio = ne_len / en_len

    if 0.35 <= ratio <= 3.0:
        return ratio, "ok"
    if 0.20 <= ratio < 0.35 or 3.0 < ratio <= 4.5:
        return ratio, "warning"
    return ratio, "bad"


def digit_signature(text: str) -> tuple[str, ...]:
    normalized = str(text).translate(str.maketrans("०१२३४५६७८९", "0123456789"))
    return tuple(re.findall(r"\d+(?:\.\d+)?", normalized))


def is_bullet_like(text: str) -> bool:
    text = clean_for_alignment(text)
    return text.startswith("•") or bool(re.match(r"^\d+\.", text))


def pair_score(en_text: str, ne_text: str) -> float:
    ratio, ratio_status = length_ratio_flag(en_text, ne_text)
    score_map = {
        "ok": 4.0,
        "warning": 1.5,
        "bad": -3.0,
    }

    score = score_map[ratio_status]

    if is_bullet_like(en_text) == is_bullet_like(ne_text):
        score += 0.75
    else:
        score -= 0.5

    if digit_signature(en_text) and digit_signature(en_text) == digit_signature(ne_text):
        score += 0.75

    if is_heading_like(en_text) or is_heading_like(ne_text):
        score -= 1.0

    if is_too_short(en_text, "en") or is_too_short(ne_text, "ne"):
        score -= 1.5

    if len(en_text) > 0:
        score -= abs(1.0 - ratio) * 0.25

    return score


def skip_cost(text: str, lang: str) -> float:
    if is_heading_like(text):
        return 0.75

    if is_too_short(text, lang):
        return 1.0

    if is_bullet_like(text):
        return 2.0

    return 2.25


def combined_text(records: list[dict], start: int, size: int) -> str:
    return " ".join(record["sentence"] for record in records[start:start + size]).strip()


def span_pair_score(en_records: list[dict], en_index: int, en_size: int, ne_records: list[dict], ne_index: int, ne_size: int) -> float:
    en_text = combined_text(en_records, en_index, en_size)
    ne_text = combined_text(ne_records, ne_index, ne_size)
    penalty = 0.4 * max(0, (en_size + ne_size) - 2)
    return pair_score(en_text, ne_text) - penalty


def can_use_multi_span(records: list[dict], start: int, size: int, lang: str) -> bool:
    if size <= 1:
        return True

    for record in records[start:start + size]:
        text = record["sentence"]
        if is_too_short(text, lang) or is_heading_like(text) or is_urlish_fragment(text):
            return True
        if is_bullet_like(text) and len(text.replace("•", " ").split()) <= 6:
            return True

    return False


def assign_alignment_quality(en: str, ne: str, ratio_status: str, score: float) -> str:
    if is_too_short(en, "en") or is_too_short(ne, "ne"):
        return "needs_review_short"

    if is_heading_like(en) or is_heading_like(ne):
        return "needs_review_heading"

    if ratio_status == "bad":
        return "needs_review_length_mismatch"

    if ratio_status == "warning" or score < 2.5:
        return "medium_confidence"

    return "high_confidence"


def build_runs(records: list[dict]) -> list[dict]:
    runs = []
    current_run = None

    for record in records:
        language = record["language"]

        if current_run and current_run["language"] == language:
            current_run["records"].append(record)
            continue

        current_run = {
            "language": language,
            "records": [record],
        }
        runs.append(current_run)

    return runs


def pair_runs(runs: list[dict]) -> tuple[list[dict], int, int]:
    run_pairs = []
    unmatched_en = 0
    unmatched_ne = 0
    index = 0

    while index < len(runs):
        current = runs[index]
        current_lang = current["language"]

        if current_lang not in {"en", "ne"}:
            index += 1
            continue

        if index + 1 >= len(runs):
            if current_lang == "en":
                unmatched_en += len(current["records"])
            else:
                unmatched_ne += len(current["records"])
            index += 1
            continue

        nxt = runs[index + 1]
        next_lang = nxt["language"]

        if next_lang not in {"en", "ne"} or next_lang == current_lang:
            if current_lang == "en":
                unmatched_en += len(current["records"])
            else:
                unmatched_ne += len(current["records"])
            index += 1
            continue

        if current_lang == "en":
            run_pairs.append({"en": current["records"], "ne": nxt["records"]})
        else:
            run_pairs.append({"en": nxt["records"], "ne": current["records"]})

        index += 2

    return run_pairs, unmatched_en, unmatched_ne


def align_run_sentences(en_records: list[dict], ne_records: list[dict]) -> tuple[list[dict], int, int]:
    en_len = len(en_records)
    ne_len = len(ne_records)

    dp = [[0.0 for _ in range(ne_len + 1)] for _ in range(en_len + 1)]
    choice = [[None for _ in range(ne_len + 1)] for _ in range(en_len + 1)]

    for en_index in range(en_len - 1, -1, -1):
        dp[en_index][ne_len] = dp[en_index + 1][ne_len] - skip_cost(en_records[en_index]["sentence"], "en")
        choice[en_index][ne_len] = "skip_en"

    for ne_index in range(ne_len - 1, -1, -1):
        dp[en_len][ne_index] = dp[en_len][ne_index + 1] - skip_cost(ne_records[ne_index]["sentence"], "ne")
        choice[en_len][ne_index] = "skip_ne"

    for en_index in range(en_len - 1, -1, -1):
        for ne_index in range(ne_len - 1, -1, -1):
            options = []

            options.append(
                (
                    span_pair_score(en_records, en_index, 1, ne_records, ne_index, 1)
                    + dp[en_index + 1][ne_index + 1],
                    ("match", 1, 1),
                )
            )

            if ne_index + 1 < ne_len and can_use_multi_span(ne_records, ne_index, 2, "ne"):
                options.append(
                    (
                        span_pair_score(en_records, en_index, 1, ne_records, ne_index, 2)
                        + dp[en_index + 1][ne_index + 2],
                        ("match", 1, 2),
                    )
                )

            if en_index + 1 < en_len and can_use_multi_span(en_records, en_index, 2, "en"):
                options.append(
                    (
                        span_pair_score(en_records, en_index, 2, ne_records, ne_index, 1)
                        + dp[en_index + 2][ne_index + 1],
                        ("match", 2, 1),
                    )
                )

            skip_en_score = dp[en_index + 1][ne_index] - skip_cost(en_records[en_index]["sentence"], "en")
            skip_ne_score = dp[en_index][ne_index + 1] - skip_cost(ne_records[ne_index]["sentence"], "ne")
            options.append((skip_en_score, "skip_en"))
            options.append((skip_ne_score, "skip_ne"))

            best_score, best_choice = max(options, key=lambda item: item[0])

            dp[en_index][ne_index] = best_score
            choice[en_index][ne_index] = best_choice

    alignments = []
    en_index = 0
    ne_index = 0
    unmatched_en = 0
    unmatched_ne = 0

    while en_index < en_len or ne_index < ne_len:
        current_choice = choice[en_index][ne_index]

        if isinstance(current_choice, tuple) and current_choice[0] == "match":
            _, en_size, ne_size = current_choice
            score = span_pair_score(en_records, en_index, en_size, ne_records, ne_index, ne_size)
            alignments.append(
                {
                    "en_start": en_index,
                    "en_size": en_size,
                    "ne_start": ne_index,
                    "ne_size": ne_size,
                    "score": score,
                }
            )
            en_index += en_size
            ne_index += ne_size
        elif current_choice == "skip_en":
            unmatched_en += 1
            en_index += 1
        elif current_choice == "skip_ne":
            unmatched_ne += 1
            ne_index += 1
        else:
            break

    return alignments, unmatched_en, unmatched_ne


def main():
    rows = read_csv_rows(SENTENCE_CANDIDATES_PATH)

    missing = [column for column in INPUT_FIELDNAMES if column not in rows[0]] if rows else INPUT_FIELDNAMES
    if missing:
        raise ValueError(f"Missing columns in sentence_candidates.csv: {missing}")

    grouped_rows = {}
    source_order = []

    for row in rows:
        source_id = row["source_id"]

        if source_id not in grouped_rows:
            grouped_rows[source_id] = []
            source_order.append(source_id)

        grouped_rows[source_id].append(row)

    aligned_rows = []
    manual_review_rows = []
    log_rows = []
    pair_counter = 1

    for source_id in source_order:
        group = sorted(
            grouped_rows[source_id],
            key=lambda item: natural_sentence_id(item["global_sentence_id"]),
        )

        title = group[0]["title"]
        subdomain = group[0]["subdomain"]

        filtered_records = []
        raw_en_count = 0
        raw_ne_count = 0

        for record in group:
            language = record["language"]
            sentence = strip_header_prefixes(record["sentence"], title)

            if language == "en":
                raw_en_count += 1
            elif language == "ne":
                raw_ne_count += 1

            if language not in {"en", "ne"}:
                continue

            if is_alignment_noise(sentence, language, title):
                continue

            normalized_record = dict(record)
            normalized_record["sentence"] = sentence
            filtered_records.append(normalized_record)

        num_en = sum(1 for record in filtered_records if record["language"] == "en")
        num_ne = sum(1 for record in filtered_records if record["language"] == "ne")

        print(f"\nAligning {source_id} - {title}")
        print(f"Filtered EN: {num_en} | Filtered NE: {num_ne}")

        if num_en == 0 or num_ne == 0:
            log_rows.append(
                {
                    "source_id": source_id,
                    "title": title,
                    "subdomain": subdomain,
                    "num_en_sentences": raw_en_count,
                    "num_ne_sentences": raw_ne_count,
                    "num_aligned_pairs": 0,
                    "num_unmatched_en": raw_en_count,
                    "num_unmatched_ne": raw_ne_count,
                    "num_run_pairs": 0,
                    "alignment_status": "skipped_missing_parallel_language",
                }
            )
            continue

        runs = build_runs(filtered_records)
        run_pairs, unmatched_en_runs, unmatched_ne_runs = pair_runs(runs)

        aligned_count = 0
        unmatched_en = unmatched_en_runs
        unmatched_ne = unmatched_ne_runs

        for run_pair in run_pairs:
            run_alignments, run_unmatched_en, run_unmatched_ne = align_run_sentences(
                run_pair["en"],
                run_pair["ne"],
            )

            unmatched_en += run_unmatched_en
            unmatched_ne += run_unmatched_ne

            for alignment in run_alignments:
                en_slice = run_pair["en"][alignment["en_start"]:alignment["en_start"] + alignment["en_size"]]
                ne_slice = run_pair["ne"][alignment["ne_start"]:alignment["ne_start"] + alignment["ne_size"]]

                en_text = " ".join(row["sentence"] for row in en_slice)
                ne_text = " ".join(row["sentence"] for row in ne_slice)
                score = alignment["score"]

                ratio, ratio_status = length_ratio_flag(en_text, ne_text)
                quality = assign_alignment_quality(en_text, ne_text, ratio_status, score)
                method_suffix = f"{len(en_slice)}x{len(ne_slice)}"

                aligned_row = {
                    "pair_id": f"PAIR_{pair_counter:07d}",
                    "source_id": source_id,
                    "title": title,
                    "subdomain": subdomain,
                    "en_sentence_id": "|".join(row["global_sentence_id"] for row in en_slice),
                    "ne_sentence_id": "|".join(row["global_sentence_id"] for row in ne_slice),
                    "en_paragraph_id": "|".join(row["paragraph_id"] for row in en_slice),
                    "ne_paragraph_id": "|".join(row["paragraph_id"] for row in ne_slice),
                    "en": en_text,
                    "ne": ne_text,
                    "en_chars": len(en_text),
                    "ne_chars": len(ne_text),
                    "length_ratio_ne_en": round(ratio, 3),
                    "length_ratio_status": ratio_status,
                    "alignment_method": f"language_run_dp_v2_{method_suffix}",
                    "quality_label": quality,
                    "review_status": "pending",
                }

                aligned_rows.append(aligned_row)
                manual_review_rows.append(
                    {
                        "pair_id": aligned_row["pair_id"],
                        "source_id": source_id,
                        "title": title,
                        "subdomain": subdomain,
                        "en": en_text,
                        "ne": ne_text,
                        "quality_label": quality,
                        "length_ratio_status": ratio_status,
                        "review_status": "pending",
                        "corrected_en": "",
                        "corrected_ne": "",
                        "review_notes": "",
                    }
                )

                pair_counter += 1
                aligned_count += 1

        status = "aligned_runwise"
        if unmatched_en or unmatched_ne:
            status = "aligned_with_unmatched_sentences"

        log_rows.append(
            {
                "source_id": source_id,
                "title": title,
                "subdomain": subdomain,
                "num_en_sentences": raw_en_count,
                "num_ne_sentences": raw_ne_count,
                "num_aligned_pairs": aligned_count,
                "num_unmatched_en": unmatched_en,
                "num_unmatched_ne": unmatched_ne,
                "num_run_pairs": len(run_pairs),
                "alignment_status": status,
            }
        )

    write_csv_rows(ALIGNED_OUTPUT_PATH, ALIGNED_FIELDNAMES, aligned_rows)
    write_csv_rows(MANUAL_REVIEW_OUTPUT_PATH, MANUAL_REVIEW_FIELDNAMES, manual_review_rows)
    write_csv_rows(ALIGNMENT_LOG_PATH, LOG_FIELDNAMES, log_rows)

    print("\nAlignment complete.")
    print(f"Raw aligned pairs saved to: {ALIGNED_OUTPUT_PATH}")
    print(f"Manual review template saved to: {MANUAL_REVIEW_OUTPUT_PATH}")
    print(f"Alignment log saved to: {ALIGNMENT_LOG_PATH}")


if __name__ == "__main__":
    main()

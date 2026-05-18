import csv
import re
from pathlib import Path


CLEANING_LOG_PATH = Path("data/logs/cleaning_log.csv")
SENTENCE_OUTPUT_PATH = Path("data/sentences/sentence_candidates.csv")
SENTENCE_LOG_PATH = Path("data/logs/sentence_split_log.csv")

SENTENCE_FIELDNAMES = [
    "global_sentence_id",
    "source_id",
    "title",
    "subdomain",
    "paragraph_id",
    "local_sentence_id",
    "language",
    "sentence",
    "num_chars",
    "num_devanagari_chars",
    "num_latin_chars",
]

LOG_FIELDNAMES = [
    "source_id",
    "title",
    "subdomain",
    "num_sentences",
    "num_en",
    "num_ne",
    "num_mixed",
    "num_unknown",
    "status",
]

BULLET_CHARS = "•◦▪●·"
FIGURE_LABEL_PATTERNS = [
    "retinal blood vessels",
    "back of eye",
    "front of eye",
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


def normalize_spacing(text: str) -> str:
    text = text.replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_line(line: str) -> str:
    line = line.replace("\t", " ")
    line = line.replace("", "•")
    line = line.replace("▪", "•")
    line = line.replace("●", "•")
    line = line.replace("·", "•")
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def dedupe_consecutive_lines(lines: list[str]) -> list[str]:
    deduped = []
    previous = None

    for line in lines:
        if line and line == previous:
            continue

        deduped.append(line)

        if line:
            previous = line

    return deduped


def drop_short_label_runs(lines: list[str]) -> list[str]:
    """
    Remove dense runs of short, punctuation-free label lines.
    These usually come from embedded diagrams such as eye anatomy figures.
    """
    kept = []
    run = []

    def flush_run() -> None:
        nonlocal run

        if len(run) >= 4:
            run = []
            return

        kept.extend(run)
        run = []

    for line in lines:
        if not line:
            flush_run()
            kept.append(line)
            continue

        word_count = len(line.split())
        looks_like_label = (
            not is_bullet_marker_only(line)
            and not sentence_terminal(line)
            and 1 <= word_count <= 4
        )

        if looks_like_label:
            run.append(line)
            continue

        flush_run()
        kept.append(line)

    flush_run()
    return kept


def count_devanagari(text: str) -> int:
    return len(re.findall(r"[\u0900-\u097F]", text))


def count_latin(text: str) -> int:
    return len(re.findall(r"[A-Za-z]", text))


def detect_language(text: str) -> str:
    """
    Labels sentence as:
    - ne: mostly Devanagari/Nepali
    - en: mostly Latin/English
    - mixed: both are present significantly
    - unknown: not enough signal
    """
    deva = count_devanagari(text)
    latin = count_latin(text)
    total = deva + latin

    if total == 0:
        return "unknown"

    deva_ratio = deva / total
    latin_ratio = latin / total

    if deva >= 2 and deva_ratio >= 0.60:
        return "ne"
    if latin >= 2 and latin_ratio >= 0.60:
        return "en"
    if deva > 0 and latin > 0:
        return "mixed"

    return "unknown"


def is_bullet_marker_only(line: str) -> bool:
    return bool(re.fullmatch(rf"[{re.escape(BULLET_CHARS)}]+", line))


def normalize_bullet_line(line: str) -> str:
    content = line.lstrip(BULLET_CHARS).strip()
    return f"• {content}" if content else "•"


def is_disclaimer_line(line: str) -> bool:
    lower = line.lower()

    disclaimer_patterns = [
        "healthinfotranslations.org",
        "copyright",
        "unless otherwise stated",
        "the medical information found on this website",
        "you should always seek the advice of your doctor",
        "the ohio state university",
        "wexner medical center",
        "mount carmel health system",
        "ohiohealth",
        "nationwide children",
    ]

    return any(pattern in lower for pattern in disclaimer_patterns)


def strip_leading_page_number(line: str) -> str:
    return re.sub(r"^\d+\s+", "", line).strip()


def should_skip_line(line: str, title: str) -> bool:
    if not line:
        return True

    lower = line.lower()
    title_lower = title.lower()
    stripped_page = strip_leading_page_number(lower)

    if lower in {"nepali.", "healthinfotranslations.org"}:
        return True

    if re.fullmatch(r"\d+", line):
        return True

    if lower in {title_lower, f"{title_lower}."}:
        return True

    if lower == f"{title_lower}. nepali.":
        return True

    if stripped_page in {title_lower, f"{title_lower}.", f"{title_lower}. nepali.", "healthinfotranslations.org"}:
        return True

    if is_disclaimer_line(line):
        return True

    return False


def sentence_terminal(text: str) -> bool:
    return bool(re.search(r"[।.!?:]$", text))


def clean_sentence_text(sentence: str) -> str:
    sentence = sentence.replace("Ì", "")
    sentence = sentence.replace("�", "")
    sentence = re.sub(r"[\uE000-\uF8FF]", "", sentence)
    sentence = re.sub(r"([।.!?])(?=[^\s])", r"\1 ", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    sentence = re.sub(r"•\s+", "• ", sentence)
    sentence = collapse_adjacent_duplicate_tokens(sentence)
    return sentence.strip()


def collapse_adjacent_duplicate_tokens(text: str) -> str:
    words = text.split()

    if len(words) < 4:
        return text

    max_size = min(12, len(words) // 2)

    for size in range(max_size, 1, -1):
        collapsed = []
        index = 0
        changed = False

        while index < len(words):
            left = words[index:index + size]
            right = words[index + size:index + (2 * size)]

            if len(left) == size and left == right:
                collapsed.extend(left)
                index += 2 * size
                changed = True
            else:
                collapsed.append(words[index])
                index += 1

        if changed:
            words = collapsed

    return " ".join(words)


def is_noise_sentence(sentence: str, lang: str, title: str) -> bool:
    sentence = sentence.strip()
    lower = sentence.lower()
    word_count = len(sentence.replace("•", " ").split())

    if len(sentence) < 3:
        return True

    if lower in {title.lower(), f"{title.lower()}.", "nepali."}:
        return True

    if re.fullmatch(r"\d+", sentence):
        return True

    if is_disclaimer_line(sentence):
        return True

    if any(pattern in lower for pattern in FIGURE_LABEL_PATTERNS):
        return True

    if word_count <= 12 and not sentence.startswith("•") and not sentence_terminal(sentence):
        return True

    if lang in {"unknown", "mixed"} and word_count <= 8 and not sentence_terminal(sentence):
        return True

    if lang == "en" and count_latin(sentence) < 3:
        return True

    if lang == "ne" and count_devanagari(sentence) < 3:
        return True

    return False


def split_text_block(block_text: str, is_bullet: bool) -> list[str]:
    parts = re.split(r"(?<=[।.!?])\s+", block_text)
    parts = [part.strip() for part in parts if part.strip()]

    if not is_bullet:
        return parts

    if not parts:
        return []

    normalized_parts = []

    for index, part in enumerate(parts):
        if index == 0:
            normalized_parts.append(part)
            continue

        normalized_parts.append(part.lstrip("• ").strip())

    return [part for part in normalized_parts if part]


def build_sentence_blocks(text: str, title: str) -> list[dict]:
    lines = [normalize_line(line) for line in normalize_spacing(text).splitlines()]
    lines = dedupe_consecutive_lines(lines)
    lines = drop_short_label_runs(lines)

    blocks = []
    current_lines = []
    current_lang = "unknown"
    current_is_bullet = False
    pending_bullet = False

    def flush_current() -> None:
        nonlocal current_lines, current_lang, current_is_bullet

        if not current_lines:
            return

        block_text = clean_sentence_text(" ".join(current_lines))
        if block_text:
            blocks.append(
                {
                    "text": block_text,
                    "language": current_lang,
                    "is_bullet": current_is_bullet,
                }
            )

        current_lines = []
        current_lang = "unknown"
        current_is_bullet = False

    for raw_line in lines:
        if not raw_line:
            flush_current()
            pending_bullet = False
            continue

        if should_skip_line(raw_line, title):
            flush_current()
            pending_bullet = False
            continue

        if is_bullet_marker_only(raw_line):
            flush_current()
            pending_bullet = True
            continue

        line = raw_line
        line_is_bullet = False

        if pending_bullet:
            line = normalize_bullet_line(line)
            line_is_bullet = True
            pending_bullet = False
        elif line and line[0] in BULLET_CHARS:
            line = normalize_bullet_line(line)
            line_is_bullet = True

        line_lang = detect_language(line)

        if current_lines:
            strong_language_switch = (
                current_lang in {"en", "ne"}
                and line_lang in {"en", "ne"}
                and current_lang != line_lang
            )

            current_text = current_lines[-1]
            current_ended = sentence_terminal(current_text)

            if line_is_bullet:
                flush_current()
            elif strong_language_switch:
                flush_current()
            elif current_is_bullet and current_ended:
                flush_current()
            elif current_ended and (line_is_bullet or line_lang in {"mixed", "unknown"}):
                flush_current()

        if not current_lines:
            current_lang = line_lang
            current_is_bullet = line_is_bullet

        current_lines.append(line)

        if not current_is_bullet and sentence_terminal(line):
            flush_current()

    flush_current()
    return blocks


def main():
    rows = read_csv_rows(CLEANING_LOG_PATH)

    sentence_rows = []
    log_rows = []
    global_sentence_id = 1

    for index, row in enumerate(rows, start=1):
        source_id = row["source_id"]
        title = row["title"]
        subdomain = row["subdomain"]
        cleaned_text_path = row["cleaned_text_path"]
        cleaning_status = row["cleaning_status"]

        print(f"\n[{index}/{len(rows)}] Splitting: {source_id} - {title}")

        if cleaning_status != "cleaned":
            log_rows.append(
                {
                    "source_id": source_id,
                    "title": title,
                    "subdomain": subdomain,
                    "num_sentences": 0,
                    "num_en": 0,
                    "num_ne": 0,
                    "num_mixed": 0,
                    "num_unknown": 0,
                    "status": "skipped_not_cleaned",
                }
            )
            continue

        input_path = Path(cleaned_text_path)

        if not input_path.exists():
            log_rows.append(
                {
                    "source_id": source_id,
                    "title": title,
                    "subdomain": subdomain,
                    "num_sentences": 0,
                    "num_en": 0,
                    "num_ne": 0,
                    "num_mixed": 0,
                    "num_unknown": 0,
                    "status": "failed_cleaned_file_not_found",
                }
            )
            continue

        try:
            text = input_path.read_text(encoding="utf-8")
            blocks = build_sentence_blocks(text, title)

            doc_sentence_count = 0
            lang_counts = {
                "en": 0,
                "ne": 0,
                "mixed": 0,
                "unknown": 0,
            }

            for paragraph_id, block in enumerate(blocks, start=1):
                sentences = split_text_block(block["text"], block["is_bullet"])
                local_sentence_id = 1

                for sentence in sentences:
                    sentence = clean_sentence_text(sentence)
                    lang = detect_language(sentence)

                    if is_noise_sentence(sentence, lang, title):
                        continue

                    if lang not in lang_counts:
                        lang = "unknown"

                    sentence_rows.append(
                        {
                            "global_sentence_id": f"SENT_{global_sentence_id:07d}",
                            "source_id": source_id,
                            "title": title,
                            "subdomain": subdomain,
                            "paragraph_id": paragraph_id,
                            "local_sentence_id": local_sentence_id,
                            "language": lang,
                            "sentence": sentence,
                            "num_chars": len(sentence),
                            "num_devanagari_chars": count_devanagari(sentence),
                            "num_latin_chars": count_latin(sentence),
                        }
                    )

                    global_sentence_id += 1
                    local_sentence_id += 1
                    doc_sentence_count += 1
                    lang_counts[lang] += 1

            log_rows.append(
                {
                    "source_id": source_id,
                    "title": title,
                    "subdomain": subdomain,
                    "num_sentences": doc_sentence_count,
                    "num_en": lang_counts["en"],
                    "num_ne": lang_counts["ne"],
                    "num_mixed": lang_counts["mixed"],
                    "num_unknown": lang_counts["unknown"],
                    "status": "split",
                }
            )

            print(
                f"Sentences: {doc_sentence_count} | "
                f"EN: {lang_counts['en']} | "
                f"NE: {lang_counts['ne']} | "
                f"Mixed: {lang_counts['mixed']} | "
                f"Unknown: {lang_counts['unknown']}"
            )

        except Exception as exc:
            log_rows.append(
                {
                    "source_id": source_id,
                    "title": title,
                    "subdomain": subdomain,
                    "num_sentences": 0,
                    "num_en": 0,
                    "num_ne": 0,
                    "num_mixed": 0,
                    "num_unknown": 0,
                    "status": f"failed:{type(exc).__name__}:{exc}",
                }
            )

    write_csv_rows(SENTENCE_OUTPUT_PATH, SENTENCE_FIELDNAMES, sentence_rows)
    write_csv_rows(SENTENCE_LOG_PATH, LOG_FIELDNAMES, log_rows)

    print("\nSentence splitting complete.")
    print(f"Sentence candidates saved to: {SENTENCE_OUTPUT_PATH}")
    print(f"Sentence split log saved to: {SENTENCE_LOG_PATH}")


if __name__ == "__main__":
    main()

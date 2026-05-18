import re
import pandas as pd
from pathlib import Path


CLEANING_LOG_PATH = Path("data/logs/cleaning_log.csv")
SENTENCE_OUTPUT_PATH = Path("data/sentences/sentence_candidates.csv")
SENTENCE_LOG_PATH = Path("data/logs/sentence_split_log.csv")


def normalize_spacing(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraph-like blocks.
    Single line breaks inside a paragraph are converted to spaces.
    """
    blocks = re.split(r"\n\s*\n", text)

    paragraphs = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Join broken PDF lines inside the paragraph
        block = re.sub(r"\s*\n\s*", " ", block)
        block = re.sub(r"\s{2,}", " ", block).strip()

        if block:
            paragraphs.append(block)

    return paragraphs


def split_sentences(paragraph: str) -> list[str]:
    """
    Simple multilingual sentence splitter.
    Supports:
    English: . ? !
    Nepali: । ? !
    """
    parts = re.split(r"(?<=[।.!?])\s+", paragraph)

    sentences = []
    for sent in parts:
        sent = sent.strip()
        sent = re.sub(r"\s{2,}", " ", sent)

        if sent:
            sentences.append(sent)

    return sentences


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

    if deva_ratio >= 0.65:
        return "ne"
    elif latin_ratio >= 0.65:
        return "en"
    elif deva > 0 and latin > 0:
        return "mixed"
    else:
        return "unknown"


def is_noise_sentence(sentence: str) -> bool:
    s = sentence.strip()

    if len(s) < 3:
        return True

    noise_patterns = [
        r"^page\s+\d+",
        r"^\d+$",
        r"^www\.",
        r"^http",
        r"health information translations",
        r"copyright",
        r"last reviewed",
    ]

    lower = s.lower()

    for pattern in noise_patterns:
        if re.search(pattern, lower):
            return True

    return False


def main():
    df = pd.read_csv(CLEANING_LOG_PATH)

    sentence_rows = []
    log_rows = []

    global_sentence_id = 1

    for index, row in df.iterrows():
        source_id = row["source_id"]
        title = row["title"]
        subdomain = row["subdomain"]
        cleaned_text_path = row["cleaned_text_path"]
        cleaning_status = row["cleaning_status"]

        print(f"\n[{index + 1}/{len(df)}] Splitting: {source_id} - {title}")

        if cleaning_status != "cleaned":
            log_rows.append({
                "source_id": source_id,
                "title": title,
                "subdomain": subdomain,
                "num_sentences": 0,
                "num_en": 0,
                "num_ne": 0,
                "num_mixed": 0,
                "num_unknown": 0,
                "status": "skipped_not_cleaned"
            })
            continue

        input_path = Path(cleaned_text_path)

        if not input_path.exists():
            log_rows.append({
                "source_id": source_id,
                "title": title,
                "subdomain": subdomain,
                "num_sentences": 0,
                "num_en": 0,
                "num_ne": 0,
                "num_mixed": 0,
                "num_unknown": 0,
                "status": "failed_cleaned_file_not_found"
            })
            continue

        try:
            text = input_path.read_text(encoding="utf-8")
            text = normalize_spacing(text)

            paragraphs = split_into_paragraphs(text)

            doc_sentence_count = 0
            lang_counts = {
                "en": 0,
                "ne": 0,
                "mixed": 0,
                "unknown": 0
            }

            for paragraph_id, paragraph in enumerate(paragraphs, start=1):
                sentences = split_sentences(paragraph)

                for local_sentence_id, sentence in enumerate(sentences, start=1):
                    sentence = sentence.strip()

                    if is_noise_sentence(sentence):
                        continue

                    lang = detect_language(sentence)

                    sentence_rows.append({
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
                        "num_latin_chars": count_latin(sentence)
                    })

                    global_sentence_id += 1
                    doc_sentence_count += 1
                    lang_counts[lang] += 1

            log_rows.append({
                "source_id": source_id,
                "title": title,
                "subdomain": subdomain,
                "num_sentences": doc_sentence_count,
                "num_en": lang_counts["en"],
                "num_ne": lang_counts["ne"],
                "num_mixed": lang_counts["mixed"],
                "num_unknown": lang_counts["unknown"],
                "status": "split"
            })

            print(
                f"Sentences: {doc_sentence_count} | "
                f"EN: {lang_counts['en']} | "
                f"NE: {lang_counts['ne']} | "
                f"Mixed: {lang_counts['mixed']} | "
                f"Unknown: {lang_counts['unknown']}"
            )

        except Exception as e:
            log_rows.append({
                "source_id": source_id,
                "title": title,
                "subdomain": subdomain,
                "num_sentences": 0,
                "num_en": 0,
                "num_ne": 0,
                "num_mixed": 0,
                "num_unknown": 0,
                "status": f"failed:{type(e).__name__}:{e}"
            })

    sentence_df = pd.DataFrame(sentence_rows)
    sentence_df.to_csv(SENTENCE_OUTPUT_PATH, index=False)

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(SENTENCE_LOG_PATH, index=False)

    print("\nSentence splitting complete.")
    print(f"Sentence candidates saved to: {SENTENCE_OUTPUT_PATH}")
    print(f"Sentence split log saved to: {SENTENCE_LOG_PATH}")


if __name__ == "__main__":
    main()
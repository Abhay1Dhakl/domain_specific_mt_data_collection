import re
import unicodedata
import pandas as pd
from pathlib import Path


EXTRACTION_LOG_PATH = Path("data/logs/extraction_log.csv")
CLEANED_DIR = Path("data/cleaned/text")
CLEANING_LOG_PATH = Path("data/logs/cleaning_log.csv")


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode without destroying Nepali characters.
    NFC is safer for Devanagari text than aggressive ASCII normalization.
    """
    return unicodedata.normalize("NFC", text)


def remove_page_markers(text: str) -> str:
    """
    Remove markers added during PDF extraction like:
    --- PAGE 1 ---
    """
    text = re.sub(r"\n?\s*---\s*PAGE\s+\d+\s*---\s*\n?", "\n", text, flags=re.IGNORECASE)
    return text


def basic_cleanup(text: str) -> str:
    """
    Light cleaning only.
    Do not over-clean because we still need English and Nepali alignment later.
    """
    text = text.replace("\r", "\n")
    text = text.replace("\t", " ")
    text = text.replace("\u00a0", " ")

    # Remove zero-width characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

    # Fix broken English hyphenation across lines, e.g. "medi-\ncine" -> "medicine"
    text = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[A-Za-z])", "", text)

    # Remove extra spaces around line breaks
    text = re.sub(r"[ ]+\n", "\n", text)
    text = re.sub(r"\n[ ]+", "\n", text)

    # Collapse multiple spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    # Collapse too many blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def remove_repeated_noise_lines(text: str) -> str:
    """
    Removes very common repeated headers/footers if they occur many times.
    This is conservative: it only removes short lines repeated at least 3 times.
    """
    lines = [line.strip() for line in text.splitlines()]

    line_counts = {}
    for line in lines:
        if not line:
            continue
        if len(line) <= 80:
            line_counts[line] = line_counts.get(line, 0) + 1

    noisy_lines = {
        line for line, count in line_counts.items()
        if count >= 3 and (
            "health information translations" in line.lower()
            or "www." in line.lower()
            or "page" in line.lower()
        )
    }

    cleaned_lines = []
    for line in lines:
        if line.strip() in noisy_lines:
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def clean_text(text: str) -> str:
    text = normalize_unicode(text)
    text = remove_page_markers(text)
    text = basic_cleanup(text)
    text = remove_repeated_noise_lines(text)
    text = basic_cleanup(text)
    return text


def main():
    df = pd.read_csv(EXTRACTION_LOG_PATH)

    logs = []

    for index, row in df.iterrows():
        source_id = row["source_id"]
        title = row["title"]
        subdomain = row["subdomain"]
        text_path = row["text_path"]
        extraction_status = row["extraction_status"]

        print(f"\n[{index + 1}/{len(df)}] Cleaning: {source_id} - {title}")

        if extraction_status != "extracted":
            logs.append({
                "source_id": source_id,
                "title": title,
                "subdomain": subdomain,
                "input_text_path": text_path,
                "cleaned_text_path": "",
                "original_chars": 0,
                "cleaned_chars": 0,
                "cleaning_status": "skipped_not_extracted"
            })
            continue

        input_path = Path(text_path)

        if not input_path.exists():
            logs.append({
                "source_id": source_id,
                "title": title,
                "subdomain": subdomain,
                "input_text_path": str(input_path),
                "cleaned_text_path": "",
                "original_chars": 0,
                "cleaned_chars": 0,
                "cleaning_status": "failed_text_file_not_found"
            })
            continue

        try:
            raw_text = input_path.read_text(encoding="utf-8")
            cleaned = clean_text(raw_text)

            output_dir = CLEANED_DIR / subdomain
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / f"{source_id}.txt"
            output_path.write_text(cleaned, encoding="utf-8")

            logs.append({
                "source_id": source_id,
                "title": title,
                "subdomain": subdomain,
                "input_text_path": str(input_path),
                "cleaned_text_path": str(output_path),
                "original_chars": len(raw_text),
                "cleaned_chars": len(cleaned),
                "cleaning_status": "cleaned"
            })

            print(f"Cleaned file saved to: {output_path}")

        except Exception as e:
            logs.append({
                "source_id": source_id,
                "title": title,
                "subdomain": subdomain,
                "input_text_path": str(input_path),
                "cleaned_text_path": "",
                "original_chars": 0,
                "cleaned_chars": 0,
                "cleaning_status": f"failed:{type(e).__name__}:{e}"
            })

    log_df = pd.DataFrame(logs)
    CLEANING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(CLEANING_LOG_PATH, index=False)

    print("\nCleaning complete.")
    print(f"Cleaning log saved to: {CLEANING_LOG_PATH}")


if __name__ == "__main__":
    main()
import re
import time
import requests
import pandas as pd
from pathlib import Path

registry_path = Path("data/source_registry.csv")
raw_pdf_path = Path("data/raw/pdfs")
log_path = Path("data/logs/download_log.csv")

required_columns = [
    "source_id",
    "source_name",
    "url",
    "title",
    "subdomain",
    "data_type",
    "language_type",
    "file_type",
    "access_date",
    "license_note",
    "status",
]

def slugify(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def validate_registry(df: pd.DataFrame) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in source_registry.csv: {missing}")


def download_pdf(url: str, output_path: Path) -> str:
    headers = {
        "User-Agent": "NepaliHealthcareMTResearch/0.1"
    }

    try:
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        if "pdf" not in content_type and not url.lower().endswith(".pdf"):
            return f"skipped_non_pdf:{content_type}"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(response.content)

        if output_path.stat().st_size == 0:
            return "failed_empty_file"

        return "downloaded"

    except Exception as e:
        return f"failed:{type(e).__name__}:{e}"


def main():
    df = pd.read_csv(registry_path)
    validate_registry(df)

    logs = []
    updated_status = []

    for index, row in df.iterrows():
        source_id = row["source_id"]
        url = row["url"]
        title = row["title"]
        subdomain = row["subdomain"]

        clean_title = slugify(title)
        clean_subdomain = slugify(subdomain)

        filename = f"{source_id}_{clean_title}.pdf"
        output_path = raw_pdf_path / clean_subdomain / filename

        print(f"\n[{index + 1}/{len(df)}] {source_id}: {title}")

        if output_path.exists() and output_path.stat().st_size > 0:
            status = "already_downloaded"
            print(f"Already exists: {output_path}")
        else:
            print(f"Downloading from: {url}")
            status = download_pdf(url, output_path)
            print(f"Status: {status}")

            # polite delay
            time.sleep(2)

        updated_status.append(status)

        logs.append({
            "source_id": source_id,
            "source_name": row["source_name"],
            "title": title,
            "subdomain": subdomain,
            "url": url,
            "local_path": str(output_path),
            "download_status": status,
        })

    # Update only the status column in your registry
    df["status"] = updated_status
    df.to_csv(registry_path, index=False)

    # Save separate download log with local paths
    log_df = pd.DataFrame(logs)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(log_path, index=False)

    print("\nDownload complete.")
    print(f"Updated registry: {registry_path}")
    print(f"Download log saved: {log_path}")

if __name__ == "__main__":
    main()
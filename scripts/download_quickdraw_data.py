import sys
from pathlib import Path
from urllib.parse import quote

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_NAMES, QUICKDRAW_BASE_URL, RAW_DATA_DIR


CHUNK_SIZE = 1024 * 1024
PROGRESS_STEP_MB = 25


def quickdraw_url(class_name):
    file_name = f"{class_name}.npy"
    return f"{QUICKDRAW_BASE_URL}/{quote(file_name)}"


def download_class_file(class_name):
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    file_path = RAW_DATA_DIR / f"{class_name}.npy"
    temp_path = file_path.with_suffix(".npy.part")

    if file_path.exists():
        print(f"[skip] {file_path.name} already exists.")
        return True

    print(f"[download] {class_name} -> {file_path}")

    try:
        with requests.get(quickdraw_url(class_name), stream=True, timeout=60) as response:
            response.raise_for_status()

            total_bytes = int(response.headers.get("content-length", 0))
            downloaded_bytes = 0
            next_progress_bytes = PROGRESS_STEP_MB * CHUNK_SIZE

            with temp_path.open("wb") as output_file:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue

                    output_file.write(chunk)
                    downloaded_bytes += len(chunk)

                    if downloaded_bytes >= next_progress_bytes:
                        print_progress(downloaded_bytes, total_bytes)
                        next_progress_bytes += PROGRESS_STEP_MB * CHUNK_SIZE

        temp_path.replace(file_path)
        print(f"[done] Saved {file_path.name}")
        return True

    except requests.RequestException as error:
        print(f"[error] Could not download {class_name}: {error}")
    except OSError as error:
        print(f"[error] Could not save {class_name}: {error}")

    if temp_path.exists():
        temp_path.unlink()

    return False


def print_progress(downloaded_bytes, total_bytes):
    downloaded_mb = downloaded_bytes / CHUNK_SIZE

    if total_bytes:
        total_mb = total_bytes / CHUNK_SIZE
        print(f"  {downloaded_mb:.0f} MB of {total_mb:.0f} MB downloaded")
    else:
        print(f"  {downloaded_mb:.0f} MB downloaded")


def main():
    print("Downloading Quick, Draw! NumPy bitmap files...")
    print(f"Saving files to: {RAW_DATA_DIR}")

    failed_classes = []

    for index, class_name in enumerate(CLASS_NAMES, start=1):
        print(f"\nClass {index}/{len(CLASS_NAMES)}: {class_name}")
        success = download_class_file(class_name)

        if not success:
            failed_classes.append(class_name)

    if failed_classes:
        print("\nSome downloads failed:")
        for class_name in failed_classes:
            print(f"- {class_name}")
        raise SystemExit(1)

    print("\nAll selected Quick, Draw! files are ready.")


if __name__ == "__main__":
    main()

import csv
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "video_links.csv"
OUTPUT_FILE = DATA_DIR / "youtube_comments.csv"
TRACKING_FILE = DATA_DIR / "collected_comment_ids.txt"

# Collection settings
COMMENTS_PER_PAGE = 100
MAX_COMMENTS_PER_VIDEO = None  # None = collect all available comments

# Request pacing
PAGE_DELAY_SECONDS = 1.0
VIDEO_DELAY_SECONDS = 1.5

# Automatic retry
API_MAX_RETRIES = 5

# Exact output schema for ML dataset
OUTPUT_COLUMNS = [
    "VideoTitle",
    "CommentText",
    "Sentiment",
    "Sentiment_label",
    "clean_comment_text",
    "tokens_comment_text",
    "clean_title",
    "token_title",
]


# ============================================================
# VIDEO URL HANDLING
# ============================================================


def extract_video_id(url):
    """Extract YouTube video ID from supported URL formats."""

    patterns = [
        r"(?:youtube\.com/watch\?.*?v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)

        if match:
            return match.group(1)

    return None


# ============================================================
# CSV AND TRACKING
# ============================================================


def load_video_links():
    """Load input URLs and their collection statuses."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as file:

        reader = csv.DictReader(file)

        if not {"url", "status"}.issubset(reader.fieldnames or []):
            raise ValueError("video_links.csv must contain 'url' and 'status' columns.")

        return list(reader)


def update_video_status(rows):
    """Persist collection status after each successful video."""

    temp_file = INPUT_FILE.with_suffix(".tmp")

    with open(temp_file, "w", encoding="utf-8-sig", newline="") as file:

        writer = csv.DictWriter(file, fieldnames=["url", "status"])

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "url": row.get("url", ""),
                    "status": row.get("status", ""),
                }
            )

    temp_file.replace(INPUT_FILE)


def load_existing_comment_ids():
    """Load previously collected comment IDs."""

    if not TRACKING_FILE.exists():
        return set()

    with open(TRACKING_FILE, "r", encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip()}


def save_comment_ids(comment_ids):
    """Persist IDs to prevent duplicate collection."""

    if not comment_ids:
        return

    with open(TRACKING_FILE, "a", encoding="utf-8") as file:
        for comment_id in comment_ids:
            file.write(comment_id + "\n")

        file.flush()
        os.fsync(file.fileno())


def save_comments(comments):
    """Append raw comments to the ML dataset CSV."""

    if not comments:
        return

    file_exists = OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0

    with open(OUTPUT_FILE, "a", encoding="utf-8-sig", newline="") as file:

        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)

        if not file_exists:
            writer.writeheader()

        writer.writerows(comments)

        file.flush()
        os.fsync(file.fileno())


# ============================================================
# YOUTUBE API
# ============================================================


def create_youtube_client():
    """Initialize YouTube Data API v3 client."""

    if not API_KEY:
        raise ValueError("Missing YOUTUBE_API_KEY. Check your .env file.")

    return build("youtube", "v3", developerKey=API_KEY, cache_discovery=False)


def get_video_title(youtube, video_id):
    """Retrieve video title."""

    response = (
        youtube.videos()
        .list(part="snippet", id=video_id)
        .execute(num_retries=API_MAX_RETRIES)
    )

    items = response.get("items", [])

    if not items:
        raise ValueError("Video not found or unavailable.")

    return items[0]["snippet"]["title"]


# ============================================================
# COMMENT COLLECTION
# ============================================================


def collect_comments(youtube, video_id, video_title, existing_ids):
    """Collect all available public top-level comments."""

    comments = []
    new_comment_ids = []

    next_page_token = None

    while True:

        response = (
            youtube.commentThreads()
            .list(
                part="snippet",
                videoId=video_id,
                maxResults=COMMENTS_PER_PAGE,
                pageToken=next_page_token,
                textFormat="plainText",
                order="time",
            )
            .execute(num_retries=API_MAX_RETRIES)
        )

        for item in response.get("items", []):

            thread = item["snippet"]
            comment = thread["topLevelComment"]
            snippet = comment["snippet"]

            comment_id = comment["id"]

            if comment_id in existing_ids:
                continue

            # textDisplay is returned as plain text because
            # textFormat="plainText" is specified above.
            comment_text = snippet.get("textDisplay", snippet.get("textOriginal", ""))

            comments.append(
                {
                    "VideoTitle": video_title,
                    "CommentText": comment_text,
                    "Sentiment": "",
                    "Sentiment_label": "",
                    "clean_comment_text": "",
                    "tokens_comment_text": "",
                    "clean_title": "",
                    "token_title": "",
                }
            )

            new_comment_ids.append(comment_id)
            existing_ids.add(comment_id)

            if (
                MAX_COMMENTS_PER_VIDEO is not None
                and len(comments) >= MAX_COMMENTS_PER_VIDEO
            ):
                return comments, new_comment_ids

        next_page_token = response.get("nextPageToken")

        if not next_page_token:
            break

        # Respect request pacing between pages.
        time.sleep(PAGE_DELAY_SECONDS)

    return comments, new_comment_ids


# ============================================================
# ERROR HANDLING
# ============================================================


def is_quota_exceeded(error):
    """Check whether Google API quota has been exhausted."""

    content = getattr(error, "content", b"")

    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="ignore")

    return "quotaExceeded" in content


def is_comments_unavailable(error):
    """Identify videos where comments cannot be retrieved."""

    content = getattr(error, "content", b"")

    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="ignore")

    return any(
        reason in content
        for reason in (
            "commentsDisabled",
            "forbidden",
        )
    )


# ============================================================
# MAIN
# ============================================================


def main():

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    youtube = create_youtube_client()

    rows = load_video_links()

    existing_ids = load_existing_comment_ids()

    total_videos = len(rows)
    collected_videos = 0
    failed_videos = 0

    print("=" * 55)
    print("       YOUTUBE COMMENT COLLECTOR")
    print("=" * 55)

    print(f"Total videos: {total_videos}")
    print(f"Previously collected comments: {len(existing_ids)}")

    for index, row in enumerate(rows):

        url = row.get("url", "").strip()
        status = row.get("status", "").strip()

        # Skip empty or already collected entries.
        if not url or status.lower() == "collected":
            continue

        video_id = extract_video_id(url)

        if not video_id:
            print(f"\n[INVALID URL] {url}")
            failed_videos += 1
            continue

        print(f"\n[{index + 1}/{total_videos}] Video: {video_id}")

        try:

            # Step 1: Get video metadata.
            video_title = get_video_title(youtube, video_id)

            print(f"Title: {video_title}")

            # Step 2: Collect comments.
            comments, new_ids = collect_comments(
                youtube, video_id, video_title, existing_ids
            )

            # Step 3: Save raw comments and tracking IDs.
            save_comments(comments)
            save_comment_ids(new_ids)

            # Step 4: Update status only after successful completion.
            row["status"] = "Collected"
            update_video_status(rows)

            collected_videos += 1

            print(f"[SUCCESS] Collected {len(comments)} new comments.")

        except HttpError as error:

            if is_quota_exceeded(error):
                print("\n[STOP] Daily YouTube API quota exhausted.")
                print("Resume collection after quota becomes available.")
                break

            if is_comments_unavailable(error):
                print(f"[UNAVAILABLE] Comments cannot be retrieved " f"for {video_id}.")
            else:
                print(f"[API ERROR] {error}")

            failed_videos += 1

        except Exception as error:

            print(f"[ERROR] {video_id}: {error}")
            failed_videos += 1

        # Delay between videos.
        time.sleep(VIDEO_DELAY_SECONDS)

    print("\n" + "=" * 55)
    print("COLLECTION SUMMARY")
    print("=" * 55)

    print(f"Total videos: {total_videos}")
    print(f"Successfully collected: {collected_videos}")
    print(f"Failed or unavailable: {failed_videos}")
    print(f"Output: {OUTPUT_FILE}")

    print("\nCollection finished.")


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Collection stopped by user.")

    except Exception as error:
        print(f"\n[FATAL ERROR] {error}")
        sys.exit(1)

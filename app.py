"""FastAPI mock backend for the Creator Insights dashboard.

Run with: uvicorn app:app --reload
"""
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Creator Insights API", version="0.1.0")
BASE = {"thumbnail": "https://images.unsplash.com/photo-1492619375914-88005aa9e8fb?w=800&q=80", "status": "Completed"}
VIDEOS: list[dict[str, Any]] = [
    {**BASE, "id": "v1", "title": "Understanding Artificial Intelligence", "thumbnail": "https://images.unsplash.com/photo-1518770660439-4636190af475?w=800&q=80", "comments": 2400, "positive": 82, "neutral": 13, "negative": 5, "analyzed": "Today, 09:42 AM", "topics": ["Clear explanation", "AI basics", "Examples", "Audio quality"]},
    {**BASE, "id": "v2", "title": "My Programming Journey", "thumbnail": "https://images.unsplash.com/photo-1515879218367-8466d910aaa4?w=800&q=80", "comments": 1850, "positive": 62, "neutral": 23, "negative": 15, "analyzed": "Yesterday, 04:15 PM", "topics": ["Career advice", "Learning path", "Motivation", "Pacing"]},
    {**BASE, "id": "v3", "title": "Technology Trends in Cambodia", "thumbnail": "https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=800&q=80", "comments": 3100, "positive": 45, "neutral": 23, "negative": 32, "analyzed": "Oct 30, 2024", "topics": ["Local context", "Data accuracy", "Future jobs", "Length"]},
    {**BASE, "id": "v4", "title": "Building a Portfolio That Gets Noticed", "comments": 5490, "positive": 76, "neutral": 17, "negative": 7, "status": "Processing", "analyzed": "Started 2 min ago", "topics": ["Portfolio tips", "Design", "Projects", "Examples"]},
]


class VideoUrls(BaseModel):
    urls: list[str]


def dashboard() -> dict[str, Any]:
    completed = [v for v in VIDEOS if v["status"] == "Completed"]
    comments = sum(v["comments"] for v in completed)
    positive = round(sum(v["comments"] * v["positive"] for v in completed) / comments, 1)
    return {"videos": VIDEOS, "video_count": len(completed), "comments": comments, "positive": positive, "positive_change": 8.2, "distribution": {"Positive": 68.5, "Neutral": 21.0, "Negative": 10.5}, "trend": {"labels": ["W1", "W2", "W3", "W4", "W5", "W6"], "positive": [61, 64, 63, 68, 66, 72], "neutral": [25, 23, 24, 20, 22, 18], "negative": [14, 13, 13, 12, 12, 10]}}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "mock"}


@app.get("/api/dashboard")
def get_dashboard() -> dict[str, Any]:
    return dashboard()


@app.get("/api/videos")
def get_videos() -> list[dict[str, Any]]:
    return VIDEOS


@app.get("/api/videos/{video_id}")
def get_video(video_id: str) -> dict[str, Any]:
    video = next((v for v in VIDEOS if v["id"] == video_id), None)
    if not video:
        raise HTTPException(404, "Video not found")
    return {**video, "comments_sample": [{"sentiment": "Positive", "text": "The explanation was really clear. I finally understand the basics."}, {"sentiment": "Neutral", "text": "Would love to see a follow-up with a real project."}, {"sentiment": "Negative", "text": "The audio gets a little difficult to hear near the end."}]}


@app.post("/api/videos", status_code=201)
def add_videos(payload: VideoUrls) -> dict[str, Any]:
    added = []
    for index, url in enumerate(dict.fromkeys(payload.urls)):
        if "youtube.com" not in url and "youtu.be" not in url:
            continue
        video = {**BASE, "id": f"new-{len(VIDEOS)}", "title": f"New YouTube video {index + 1}", "comments": 0, "positive": 0, "neutral": 0, "negative": 0, "status": "Processing", "analyzed": "Queued just now", "topics": [], "url": url}
        VIDEOS.append(video)
        added.append(video)
    return {"added": len(added), "videos": added}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse("static/index.html")

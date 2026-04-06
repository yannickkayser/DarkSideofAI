import yt_dlp
import json

def download_video_with_metadata(url, output_path="%(title)s.%(ext)s"):
    """Download a YouTube video and extract its metadata."""
    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "outtmpl": output_path,  # Output filename template
        "format": "bestvideo+bestaudio/best",  # Best quality
        "merge_output_format": "mp4",  # Merge into mp4
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)  # download=True now

    return {
        "title": info.get("title"),
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
        "duration": info.get("duration"),
        "upload_date": info.get("upload_date"),
        "channel": info.get("channel"),
        "tags": info.get("tags", []),
        "description": info.get("description"),
    }

video = download_video_with_metadata("https://www.youtube.com/watch?v=lBokZmnbcVw")
print(json.dumps(video, indent=2))
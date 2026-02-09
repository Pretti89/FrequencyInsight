# Frequency Insight (Gradio) — Render Deployment

## Deploy on Render (Docker)
1. Create a new **Web Service** on Render.
2. Choose **Build and deploy from a Git repository**.
3. Select **Docker** as the environment (Render will detect the Dockerfile).
4. Set the service to **Web Service** and deploy.

Render will automatically set the `PORT` environment variable. The app binds to `0.0.0.0:$PORT`.

## Notes
- This project uses `ffmpeg` (installed via Docker) for audio decoding.
- `yt-dlp` may be blocked by YouTube bot protections for some links; uploading audio always works.

#!/usr/bin/env python3
"""Image Topic Platform - Backend"""
import os, json, uuid, asyncio, httpx, base64
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
import shutil

app = FastAPI(title="Image Topic Platform")

# ---- Background cleanup (24h) ----
async def cleanup_old_sessions():
    """Remove sessions and uploads older than CLEANUP_HOURS."""
    while True:
        await asyncio.sleep(3600)  # check every hour
        cutoff = datetime.now(timezone.utc).timestamp() - CLEANUP_HOURS * 3600
        expired = [sid for sid, s in sessions.items() if _ts(s) < cutoff]
        for sid in expired:
            # Remove upload dir
            p = UPLOAD_DIR / sid
            if p.exists():
                shutil.rmtree(p)
            del sessions[sid]
        if expired:
            print(f"[cleanup] removed {len(expired)} expired sessions")

def _ts(s: dict) -> float:
    try: return datetime.fromisoformat(s["created_at"]).timestamp()
    except: return 0

@app.on_event("startup")
async def startup_cleanup():
    asyncio.create_task(cleanup_old_sessions())
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR.mkdir(exist_ok=True)

# --- config (from env) ---
def get_api_key():
    return os.environ.get("IMAGE_API_KEY", "")
def get_api_base():
    return os.environ.get("IMAGE_API_BASE", "https://hk-intra-paas.transsion.com/tranai-proxy/v1")
def build_headers():
    return {"Authorization": f"Bearer {get_api_key()}", "Content-Type": "application/json"}

# In-memory sessions
sessions: dict = {}
CLEANUP_HOURS = 24

class SessionCreate(BaseModel):
    user_name: str = "Anonymous"
class ConfirmTopics(BaseModel):
    confirmed_topics: list[str] = []

# ---- AI Vision ----
async def analyze_image(image_path: str) -> dict:
    api_key = get_api_key()
    if not api_key:
        return {"width": 0, "height": 0, "tags": ["no_key"], "description": "No API key", "style": "unknown", "colors": [], "scene": "unknown"}
    async with httpx.AsyncClient(timeout=60) as c:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = Path(image_path).suffix.lower()
        mime = {".jpg": "jpeg", ".png": "png", ".webp": "webp", ".gif": "gif"}.get(ext, "jpeg")
        data_url = f"data:image/{mime};base64,{b64}"
        resp = await c.post(
            f"{get_api_base()}/chat/completions",
            headers=build_headers(),
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": (
                        "Analyze this image and return JSON ONLY (no markdown, no explanation):\n"
                        '{"tags": ["tag1","tag2","tag3"], "description": "brief English description 1 sentence", '
                        '"style": "photo/illustration/painting/3d/etc", '
                        '"scene": "nature/city/portrait/food/animal/architecture/interior/abstract/other", '
                        '"colors": ["#hex1","#hex2","#hex3"], '
                        '"mood": "serene/energetic/dramatic/warm/cool/other"}'
                    )},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]}],
                "max_tokens": 300, "temperature": 0.1
            }
        )
        if resp.status_code != 200:
            return {"width": 0, "height": 0, "tags": ["api_error"], "description": "API error", "style": "unknown", "colors": [], "scene": "unknown"}
        try:
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(content)
        except:
            return {"width": 0, "height": 0, "tags": ["parse_error"], "description": content[:100], "style": "unknown", "colors": [], "scene": "unknown"}

# ---- Clustering (non-exclusive) ----
def cluster_topics(analyses: list) -> list:
    scene_counts = Counter()
    tag_counts = Counter()
    style_counts = Counter()
    for a in analyses:
        scene = a.get("scene", "other")
        if scene and scene != "other": scene_counts[scene] += 1
        for tag in a.get("tags", []): tag_counts[tag.lower()] += 1
        style = a.get("style", "unknown")
        if style and style != "unknown": style_counts[style] += 1

    topics = []
    for scene, count in scene_counts.most_common():
        if count >= 2:
            imgs, tags, colors, moods = [], set(), [], []
            for i, a in enumerate(analyses):
                if a.get("scene") == scene:
                    imgs.append(i); tags.update(a.get("tags", [])); colors.extend(a.get("colors", []))
                    if a.get("mood", "other") != "other": moods.append(a["mood"])
            colors = list(dict.fromkeys(colors))[:6]
            topics.append({
                "name": scene.title(), "image_indices": imgs,
                "tagline": f"{count} images \u00b7 {', '.join([t for t, _ in Counter(tags).most_common(4)])}",
                "top_tags": [t for t, _ in Counter(tags).most_common(8)],
                "colors": colors, "style": style_counts.most_common(1)[0][0] if style_counts else "photo",
                "mood": Counter(moods).most_common(1)[0][0] if moods else "neutral"
            })
    if not topics:
        for tag, count in tag_counts.most_common(5):
            if count >= 2:
                imgs, colors, moods = [], [], []
                for i, a in enumerate(analyses):
                    if tag in [t.lower() for t in a.get("tags", [])]:
                        imgs.append(i); colors.extend(a.get("colors", []))
                        if a.get("mood", "other") != "other": moods.append(a["mood"])
                colors = list(dict.fromkeys(colors))[:6]
                topics.append({
                    "name": tag.title(), "image_indices": imgs,
                    "tagline": f"{count} images \u00b7 {tag}",
                    "top_tags": list(set(t for i in imgs for t in analyses[i].get("tags", [])))[:8],
                    "colors": colors, "style": "photo",
                    "mood": Counter(moods).most_common(1)[0][0] if moods else "neutral"
                })
    topics.sort(key=lambda t: len(t["image_indices"]), reverse=True)
    return topics

# ---- Banner generation (984x390) ----
async def generate_banner(topic: dict) -> dict:
    colors = topic.get("colors", [])
    color_str = ", ".join(colors) if colors else "warm golden tones"
    tags = ", ".join(topic.get("top_tags", [])[:4])
    prompt = (
        f"A stunning horizontal banner image for a '{topic['name']}' photo gallery theme. "
        f"Theme: {tags}. Style: {topic.get('style', 'photo')}. Mood: {topic.get('mood', 'neutral')}. "
        f"Color palette: {color_str}. "
        f"Composition: wide 984x390, with negative space on the left for text overlay. "
        f"High quality digital art, suitable for web banner."
    )
    api_key = get_api_key()
    if not api_key:
        return {"error": "Image generation API not configured"}

    async with httpx.AsyncClient(timeout=180) as c:
        # Submit
        resp = await c.post(
            f"{get_api_base()}/imagen/volcengine",
            headers=build_headers(),
            json={"model": "doubao-seedream-5-0-260128", "prompt": prompt, "response_format": "url", "watermark": False, "size": "984x390"}
        )
        if resp.status_code != 200:
            return {"error": f"Submit failed: {resp.status_code} {resp.text[:200]}"}
        job_id = resp.json().get("id")
        if not job_id:
            return {"error": f"No job_id: {resp.json()}"}

        # Poll
        for _ in range(36):
            await asyncio.sleep(5)
            q = await c.get(f"{get_api_base()}/imagen/volcengine/{job_id}", headers=build_headers())
            qd = q.json()
            status = (qd.get("status") or "").lower()
            if status in ("completed", "succeeded", "success"):
                try:
                    return {"image_url": qd["result"]["oss_content"][0]["image_url"], "prompt": prompt, "size": "984x390"}
                except:
                    return {"error": f"Parse result failed: {qd}"}
            if status in ("failed",):
                return {"error": qd.get("error", "unknown")}
        return {"error": "Timeout after 180s"}

# ---- API Routes ----
@app.post("/api/session")
async def create_session(data: SessionCreate):
    sid = uuid.uuid4().hex[:12]; (UPLOAD_DIR / sid).mkdir(exist_ok=True)
    sessions[sid] = {"user_name": data.user_name, "images": [], "analyses": [], "topics": [], "created_at": datetime.now(timezone.utc).isoformat(), "status": "idle"}
    return {"session_id": sid}

@app.post("/api/{sid}/upload")
async def upload_images(sid: str, files: list[UploadFile] = File(...)):
    if sid not in sessions: raise HTTPException(404, "Session not found")
    s, d = sessions[sid], UPLOAD_DIR / sid; uploaded = []
    for f in files:
        if not f.content_type or not f.content_type.startswith("image/"): continue
        ext = Path(f.filename or "img").suffix or ".jpg"; fn = f"{uuid.uuid4().hex[:8]}{ext}"
        (d / fn).write_bytes(await f.read())
        s["images"].append({"filename": f.filename, "id": fn}); uploaded.append(fn)
    return {"uploaded": len(uploaded), "total": len(s["images"])}

@app.get("/api/{sid}/images")
async def list_images(sid: str):
    if sid not in sessions: raise HTTPException(404, "Session not found")
    s = sessions[sid]
    return {"total": len(s["images"]), "images": [{"id": i["id"], "filename": i["filename"], "url": f"/uploads/{sid}/{i['id']}"} for i in s["images"]]}

@app.get("/uploads/{sid}/{fn}")
async def serve_upload(sid: str, fn: str):
    p = UPLOAD_DIR / sid / fn
    if not p.exists(): raise HTTPException(404)
    return FileResponse(str(p))

@app.post("/api/{sid}/analyze")
async def analyze_images(sid: str):
    if sid not in sessions: raise HTTPException(404, "Session not found")
    s = sessions[sid]; s["status"] = "running"; s["analyses"] = []
    for i, img in enumerate(s["images"]):
        a = await analyze_image(img["path"]); a["image_id"] = img["id"]; a["index"] = i; s["analyses"].append(a)
    s["topics"] = cluster_topics(s["analyses"]); s["status"] = "done"
    return {"total": len(s["analyses"]), "topics": s["topics"], "analyses": s["analyses"]}

@app.get("/api/{sid}/status")
async def get_status(sid: str):
    if sid not in sessions: raise HTTPException(404, "Session not found")
    s = sessions[sid]
    return {"status": s["status"], "total_images": len(s["images"]), "analyzed_images": len(s["analyses"]), "topics": s["topics"]}

@app.get("/api/{sid}/topics")
async def get_topics(sid: str):
    if sid not in sessions: raise HTTPException(404, "Session not found")
    s = sessions[sid]
    return {"topics": s["topics"], "images": s["images"], "analyses": s["analyses"]}

@app.post("/api/{sid}/confirm-topics")
async def confirm_topics(sid: str, data: ConfirmTopics):
    if sid not in sessions: raise HTTPException(404, "Session not found")
    s = sessions[sid]
    if not s["topics"]: raise HTTPException(400, "No topics")
    results = []
    for name in data.confirmed_topics:
        topic = next((t for t in s["topics"] if t["name"] == name), None)
        if not topic: results.append({"name": name, "error": "Topic not found"}); continue
        if topic.get("banner"): results.append({"name": name, "banner": topic["banner"], "skipped": True}); continue
        r = await generate_banner(topic)
        if "error" in r: results.append({"name": name, "error": r["error"]}); continue
        banner_id = f"banner_{name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}.jpeg"
        async with httpx.AsyncClient() as dl:
            resp = await dl.get(r["image_url"])
            if resp.status_code == 200:
                bp = UPLOAD_DIR / sid / banner_id; bp.write_bytes(resp.content)
                topic["banner"] = {"id": banner_id, "url": f"/uploads/{sid}/{banner_id}", "remote_url": r["image_url"], "prompt": r["prompt"], "size": r["size"]}
                results.append({"name": name, "banner": topic["banner"]})
            else: results.append({"name": name, "error": "Download failed"})
    return {"results": results}

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    if full_path.startswith(("api/", "uploads/", "static/")): raise HTTPException(404)
    p = FRONTEND_DIR / "index.html"
    if p.exists(): return FileResponse(str(p))
    raise HTTPException(404)

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8899)

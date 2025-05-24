from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

app = FastAPI(title="Simple Object Detection API")

# Setup static files và templates
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
templates_dir = os.path.join(current_dir, "templates")

# Mount static files
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Setup templates
if os.path.exists(templates_dir):
    templates = Jinja2Templates(directory=templates_dir)

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    if os.path.exists(templates_dir):
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        return HTMLResponse("""
        <html>
            <head><title>Object Detection Demo</title></head>
            <body>
                <h1>Object Detection Demo</h1>
                <p>This is a simplified version for testing deployment.</p>
                <p>Upload functionality will be added once basic deployment works.</p>
            </body>
        </html>
        """)

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Simple app is running"}

@app.get("/api/test")
async def test_endpoint():
    return {"message": "API endpoint working", "version": "simple"}

# Vercel handler
handler = app 
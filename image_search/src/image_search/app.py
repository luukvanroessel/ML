"""A small web page for the image search.

    uv run uvicorn image_search.app:app --reload

Then open http://127.0.0.1:8000

The one idea that matters here: the model is loaded ONCE, when the first
request for a backend arrives, and then kept in memory for the life of
the process. Loading CLIP takes ~2 seconds. If we loaded it per request,
every search would cost those 2 seconds instead of ~50 ms. That gap is
most of what "serving a model" means.

FastAPI is used over Flask for three reasons: it reads the type hints on
these functions to validate input for free, it writes interactive API
docs at /docs from those same hints, and uvicorn (the server that runs
it) handles concurrent requests without extra work.
"""

from __future__ import annotations

import base64
import html
import io
import time
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from PIL import Image, UnidentifiedImageError

from .embed import BACKENDS, Backend, get_backend
from .search import rank
from .store import Index, load

LIBRARY = Path("data/library")
INDEX_DIR = Path("index")
THUMB = 220  # pixels, longest side

app = FastAPI(title="image search")

# name -> (model, index). Filled on first use, never emptied. This dict
# IS the "keep the model warm" strategy, and at this size that is all a
# model server needs to be.
_loaded: dict[str, tuple[Backend, Index]] = {}


def available() -> list[str]:
    """Backends that actually have an index built on disk."""
    return [
        name
        for name in BACKENDS
        if (INDEX_DIR / name / "meta.json").exists()
    ]


def ready(name: str) -> tuple[Backend, Index]:
    """Model + index for `name`, loading each at most once."""
    if name not in _loaded:
        index = load(INDEX_DIR / name)
        if index is None:
            raise HTTPException(404, f"no index for backend {name!r}")
        print(f"loading {name} (first request)...")
        _loaded[name] = (get_backend(name), index)
    return _loaded[name]


# --------------------------------------------------------------- pages

CSS = """
body { font: 15px system-ui, sans-serif; margin: 2rem auto; max-width: 62rem;
       padding: 0 1rem; color: #222; }
h1 { font-size: 1.3rem; }
form { display: flex; gap: .5rem; align-items: center; flex-wrap: wrap;
       background: #f5f5f5; padding: 1rem; border-radius: 8px; }
button { padding: .4rem 1rem; cursor: pointer; }
.grid { display: grid; gap: 1rem; margin-top: 1.5rem;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); }
figure { margin: 0; }
img { width: 100%; border-radius: 6px; display: block; background: #eee; }
figcaption { font-size: .8rem; margin-top: .3rem; }
.score { font-weight: 600; }
.path { color: #777; word-break: break-all; }
.note { color: #777; font-size: .85rem; }
.query { display: flex; gap: 1rem; align-items: flex-end; margin-top: 1.5rem; }
.query img { width: 150px; }
a { color: #06c; }
"""


def page(body: str) -> HTMLResponse:
    return HTMLResponse(
        "<!doctype html><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>image search</title><style>{CSS}</style>{body}"
    )


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    names = available()
    if not names:
        return page(
            "<h1>image search</h1><p>No index found. Build one first:"
            "<br><code>uv run python -m image_search.index "
            "data/library</code></p>"
        )
    options = "".join(f"<option>{n}</option>" for n in names)
    return page(
        "<h1>image search</h1>"
        # enctype=multipart/form-data is what lets a form carry a file.
        # It is also why the python-multipart package is a dependency.
        '<form action="/search" method="post" '
        'enctype="multipart/form-data">'
        '<input type="file" name="file" accept="image/*" required>'
        f'<select name="backend">{options}</select>'
        '<label>top <input type="number" name="k" value="8" min="1" '
        'max="24" style="width:4rem"></label>'
        "<button>search</button></form>"
        f'<p class=note>backends with an index: {", ".join(names)} '
        '&middot; <a href="/docs">API docs</a></p>'
    )


@app.post("/search", response_class=HTMLResponse)
async def search(
    file: UploadFile,
    backend: str = Form("clip"),
    k: int = Form(8),
) -> HTMLResponse:
    model, index = ready(backend)

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "that file is not an image")

    start = time.perf_counter()
    query = model.embed_image(image)
    embed_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    results = rank(index, query, min(k, len(index)))
    search_ms = (time.perf_counter() - start) * 1000

    cards = "".join(
        "<figure>"
        f'<img src="/thumb/{quote(path)}" loading="lazy">'
        f'<figcaption><span class=score>{score:.3f}</span><br>'
        f'<span class=path>{html.escape(path)}</span>'
        "</figcaption></figure>"
        for path, score in results
    )
    return page(
        '<h1><a href="/">&larr; image search</a></h1>'
        f'<div class=query><img src="{_data_uri(image)}">'
        f"<div><div><b>{model.name}</b></div>"
        f"<div class=note>{len(index)} images &middot; "
        f"embed {embed_ms:.0f} ms &middot; "
        f"search {search_ms:.1f} ms</div></div></div>"
        f'<div class=grid>{cards}</div>'
    )


# ------------------------------------------------------------- images


@app.get("/thumb/{name:path}")
def thumb(name: str) -> Response:
    path = (LIBRARY / name).resolve()

    # Never serve anything outside the library folder. Without this check
    # a request for /thumb/../../etc/passwd would walk straight out of it.
    # Any endpoint that turns user input into a file path needs this.
    if LIBRARY.resolve() not in path.parents or not path.is_file():
        raise HTTPException(404, "not found")
    return Response(_thumb_bytes(path), media_type="image/jpeg")


@lru_cache(maxsize=1024)
def _thumb_bytes(path: Path) -> bytes:
    """Shrink once, then serve from memory.

    Photos are often 500 KB each; a page of 8 is 4 MB of pixels the
    browser immediately throws away. Thumbnails make the page feel
    instant. The cache is keyed on path, so editing an image in place
    would serve a stale thumbnail until restart -- fine here, worth
    knowing.
    """
    image = Image.open(path).convert("RGB")
    image.thumbnail((THUMB, THUMB))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=80)
    return buffer.getvalue()


def _data_uri(image: Image.Image) -> str:
    """Inline the uploaded image into the page, so we never store it."""
    small = image.copy()
    small.thumbnail((THUMB, THUMB))
    buffer = io.BytesIO()
    small.save(buffer, "JPEG", quality=80)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{encoded}"


@app.get("/healthz")
def healthz() -> dict:
    """Is the service up, and which models are warm?

    Standard practice: something (Docker, Kubernetes, a load balancer)
    needs a cheap endpoint to poll that never touches the model.
    """
    return {
        "status": "ok",
        "indexes": available(),
        "loaded": sorted(_loaded),
    }

# app/main.py
import os, math, uuid, mimetypes
import numpy as np
import rasterio
from fastapi import FastAPI, UploadFile, Form, Query
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

mimetypes.add_type('application/geo+json', '.geojson')
mimetypes.add_type('application/json', '.geojson')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
STATIC_DIR = os.path.join(ROOT_DIR, "static")
DATA_DIR = os.path.join(BASE_DIR, "data")
TMP_DIR = os.path.join(BASE_DIR, "tmp")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

try:
    from .frontwave import run_frontwave
except ImportError:
    import sys
    if BASE_DIR not in sys.path:
        sys.path.append(BASE_DIR)
    from frontwave import run_frontwave

app = FastAPI(title="FrontWave API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok")

@app.get("/")
def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return PlainTextResponse(f"index.html not found at {index_path}", status_code=500)
    return FileResponse(index_path, media_type="text/html")

def _raster_stats(path: str) -> dict:
    with rasterio.open(path) as src:
        band = src.read(1, masked=True)
    data = np.ma.compressed(band).astype(np.float64)
    n_valid = int(data.size)
    n_total = int(band.size)
    n_nodata = int(n_total - n_valid)
    if n_valid == 0:
        return {"count": 0, "nodata_count": n_nodata, "min": None, "p05": None, "p25": None, "p50": None,
                "p75": None, "p95": None, "max": None, "mean": None, "std": None, "se": None,
                "ci95_low": None, "ci95_high": None, "range": None, "cv": None,
                "hist_bins": [], "hist_counts": []}
    q = np.percentile(data, [5, 25, 50, 75, 95])
    vmin = float(np.min(data)); vmax = float(np.max(data)); mean = float(np.mean(data))
    std = float(np.std(data, ddof=1)) if n_valid > 1 else float("nan")
    se = (std / math.sqrt(n_valid)) if n_valid > 1 else float("nan")
    counts, edges = np.histogram(data, bins=24)
    return {
        "count": n_valid, "nodata_count": n_nodata,
        "min": vmin, "p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]), "p75": float(q[3]), "p95": float(q[4]),
        "max": vmax, "mean": mean, "std": std, "se": se,
        "ci95_low": mean - 1.96 * se if n_valid > 1 else float("nan"),
        "ci95_high": mean + 1.96 * se if n_valid > 1 else float("nan"),
        "range": vmax - vmin, "cv": (100.0 * std / mean) if n_valid > 1 and mean != 0 else float("nan"),
        "hist_bins": edges.tolist(), "hist_counts": counts.tolist()
    }

# color palettes
def _palette_stops(name: str):
    name = (name or "").lower()
    if name == "viridis":
        return np.array([0, .25, .5, .75, 1.0]), np.array([[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]], dtype=float)
    if name == "plasma":
        return np.array([0, .25, .5, .75, 1.0]), np.array([[13,8,135],[126,3,168],[203,71,119],[248,149,64],[240,249,33]], dtype=float)
    if name == "magma":
        return np.array([0, .25, .5, .75, 1.0]), np.array([[0,0,4],[84,15,109],[187,55,84],[249,142,9],[252,255,164]], dtype=float)
    if name == "inferno":
        return np.array([0, .25, .5, .75, 1.0]), np.array([[0,0,3],[87,15,109],[187,55,84],[249,142,9],[252,255,164]], dtype=float)
    if name == "cividis":
        return np.array([0, .25, .5, .75, 1.0]), np.array([[0,32,77],[44,81,96],[90,123,101],[160,178,86],[255,233,69]], dtype=float)
    if name == "terrain":
        return np.array([0, .25, .5, .75, 1.0]), np.array([[0,120,0],[173,221,142],[255,255,191],[253,174,97],[215,25,28]], dtype=float)
    if name == "turbo":
        return np.array([0, .25, .5, .75, 1.0]), np.array([[48,18,59],[31,154,170],[52,209,91],[236,200,52],[220,47,2]], dtype=float)
    if name == "coolwarm":
        return np.array([0, .5, 1.0]), np.array([[59,76,192],[221,221,221],[180,4,38]], dtype=float)
    if name == "heat":
        return np.array([0.0, 0.25, 0.5, 0.75, 1.0]), np.array([[0,0,128],[0,255,255],[255,255,0],[255,128,0],[128,0,0]], dtype=float)
    if name == "gray":
        return np.array([0.0,1.0]), np.array([[0,0,0],[255,255,255]], dtype=float)
    # default
    return np.array([0.0,1.0]), np.array([[0,0,0],[255,255,255]], dtype=float)

def _apply_palette_01(x01: np.ndarray, palette: str):
    x = np.clip(x01, 0.0, 1.0)
    pos, col = _palette_stops(palette)
    idx = np.searchsorted(pos, x, side="right") - 1
    idx = np.clip(idx, 0, len(pos) - 2)
    p0 = pos[idx]; p1 = pos[idx + 1]
    a = (x - p0) / (p1 - p0 + 1e-12)
    c0 = col[idx]; c1 = col[idx + 1]
    r = ((1 - a) * c0[..., 0] + a * c1[..., 0]).astype(np.uint8)
    g = ((1 - a) * c0[..., 1] + a * c1[..., 1]).astype(np.uint8)
    b = ((1 - a) * c0[..., 2] + a * c1[..., 2]).astype(np.uint8)
    return r, g, b

def _render_png_from_tif(src_tif: str, dst_png: str, palette: str):
    with rasterio.open(src_tif) as src:
        arr = src.read(1, masked=True)
        bounds = src.bounds
    data = np.ma.filled(arr, np.nan).astype(np.float64)
    valid = np.isfinite(data)
    if valid.sum() == 0:
        h, w = data.shape
        rgba = np.zeros((4, h, w), dtype=np.uint8)
    else:
        # Auto stretch 2–98 percentiles internally
        lo = np.nanpercentile(data, 2.0)
        hi = np.nanpercentile(data, 98.0)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo = np.nanmin(data); hi = np.nanmax(data)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = 0.0, 1.0
        x01 = (data - lo) / (hi - lo)
        r, g, b = _apply_palette_01(x01, palette)
        alpha = np.where(valid, 255, 0).astype(np.uint8)
        rgba = np.stack([r, g, b, alpha], axis=0)
    h, w = rgba.shape[1], rgba.shape[2]
    profile = {"driver": "PNG", "height": h, "width": w, "count": 4, "dtype": "uint8"}
    with rasterio.open(dst_png, "w", **profile) as dst:
        dst.write(rgba)
    return [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]

def _quicklook_png(src_tif: str, dst_png: str):
    return _render_png_from_tif(src_tif, dst_png, "gray")

@app.post("/run")
async def run_process(
    csv_file: UploadFile,
    grid: float = Form(...),
    cell: float = Form(...),
    contour: float = Form(...),
    sep: str = Form(";"),
    lon_field: str = Form("lon"),
    lat_field: str = Form("lat"),
    date_field: str = Form("date"),
    id_field: str = Form("id"),
    weight_field: str = Form("weight"),
    # kriging params
    kriging_model: str = Form("ordinary"),
    variogram_model: str = Form("spherical"),
    var_sill: float | None = Form(None),
    var_range: float | None = Form(None),
    var_nugget: float | None = Form(None),
    nlags: int = Form(6),
    weight: bool = Form(False),
    drift_terms: str | None = Form(None),
):
    if sep.lower() in ("\\t", "tab", "tabs"):
        sep = "\t"

    tmp_csv = os.path.join(TMP_DIR, csv_file.filename)
    with open(tmp_csv, "wb") as f:
        f.write(await csv_file.read())

    run_id = uuid.uuid4().hex[:8]
    out_dir = os.path.join(DATA_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)

    drift = [t.strip() for t in (drift_terms or "").split(",") if t.strip()] or None

    res = run_frontwave(
        csv_path=tmp_csv,
        out_folder=out_dir,
        lon_field=lon_field, lat_field=lat_field, date_field=date_field,
        id_field=id_field, weight_field=weight_field,
        grid_cell_m=float(grid), krige_cell_m=float(cell), contour_interval=float(contour),
        sep=sep, dayfirst=True,
        kriging_model=kriging_model, variogram_model=variogram_model,
        var_sill=var_sill, var_range=var_range, var_nugget=var_nugget,
        nlags=nlags, weight=weight, drift_terms=drift
    )

    def to_url(p):
        if not p: return None
        p = os.path.abspath(p).replace("\\", "/")
        rel = os.path.relpath(p, DATA_DIR).replace("\\", "/")
        return f"/data/{rel}"

    images = {}
    def add_img(key, tif_path):
        if not tif_path or not os.path.exists(tif_path): return
        png_path = os.path.join(out_dir, f"{key}.png")
        bounds = _quicklook_png(tif_path, png_path)
        rel = os.path.relpath(png_path, DATA_DIR).replace("\\", "/")
        images[key] = {"url": f"/data/{rel}", "bounds": bounds}

    add_img("kriging",  res.get("kriging"))
    add_img("slope",    res.get("slope"))
    add_img("velocity", res.get("velocity"))

    urls = {
        "kriging": to_url(res.get("kriging")),
        "slope": to_url(res.get("slope")),
        "velocity": to_url(res.get("velocity")),
        "contours": to_url(res.get("contours")),
        "ellipse": to_url(res.get("ellipse")),
        "selected_points": to_url(res.get("selected_points")),
        "all_points": to_url(res.get("all_points")),
        "grid": to_url(res.get("grid")),
        "selected_points_geojson": to_url(res.get("selected_points_geojson")),
        "all_points_geojson": to_url(res.get("all_points_geojson")),
        "ellipse_geojson": to_url(res.get("ellipse_geojson")),
        "contours_geojson": to_url(res.get("contours_geojson")),
        "bundle_zip": to_url(res.get("zip")),
    }

    stats = {"velocity": _raster_stats(res["velocity"])} if res.get("velocity") else {"velocity": {"count": 0, "nodata_count": None}}

    meta = {
        "date_field": res.get("date_field", "date"),
        "params": {
            "grid_m": grid, "cell_m": cell, "contour": contour,
            "kriging_model": kriging_model, "variogram_model": variogram_model,
            "var_sill": var_sill, "var_range": var_range, "var_nugget": var_nugget,
            "nlags": nlags, "weight": weight, "drift_terms": drift or []
        }
    }

    return JSONResponse(content={"run_id": run_id, "urls": urls, "images": images, "stats": stats, "meta": meta})

@app.get("/render")
def render_raster(
    kind: str = Query(..., pattern="^(kriging|slope|velocity)$"),
    run_id: str = Query(...),
    palette: str = Query("gray"),
):
    tif_map = {"kriging": "kriging.tif", "slope": "slope.tif", "velocity": "velocity.tif"}
    src_tif = os.path.join(DATA_DIR, run_id, tif_map[kind])
    if not os.path.exists(src_tif):
        return JSONResponse({"error": "raster not found"}, status_code=404)
    out_png = os.path.join(DATA_DIR, run_id, f"{kind}_{palette}.png")
    bounds = _render_png_from_tif(src_tif, out_png, palette)
    rel = os.path.relpath(out_png, DATA_DIR).replace("\\", "/")
    return JSONResponse({"url": f"/data/{rel}?v={uuid.uuid4().hex[:6]}", "bounds": bounds})

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)

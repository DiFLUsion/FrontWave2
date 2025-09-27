# app/main.py
import os
import math
import uuid
import numpy as np
import rasterio
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
STATIC_DIR = os.path.join(ROOT_DIR, "static")
DATA_DIR = os.path.join(BASE_DIR, "data")
TMP_DIR = os.path.join(BASE_DIR, "tmp")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# import robusto
try:
    from .frontwave import run_frontwave
except ImportError:
    import sys
    if BASE_DIR not in sys.path:
        sys.path.append(BASE_DIR)
    from frontwave import run_frontwave

# geopandas opcional
try:
    import geopandas as gpd
except Exception:
    gpd = None

app = FastAPI(title="FrontWave API")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

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
    data = np.ma.compressed(band)
    n_valid = int(data.size)
    n_total = int(band.size)
    n_nodata = int(n_total - n_valid)
    if n_valid == 0:
        return {"count": 0, "nodata_count": n_nodata, "min": None, "p05": None, "p25": None, "p50": None,
                "p75": None, "p95": None, "max": None, "mean": None, "std": None, "se": None,
                "ci95_low": None, "ci95_high": None, "range": None, "cv": None}
    q = np.percentile(data, [5, 25, 50, 75, 95])
    vmin = float(np.min(data)); vmax = float(np.max(data)); mean = float(np.mean(data))
    std = float(np.std(data, ddof=1)) if n_valid > 1 else float("nan")
    se = (std / math.sqrt(n_valid)) if n_valid > 1 else float("nan")
    return {
        "count": n_valid, "nodata_count": n_nodata,
        "min": vmin, "p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]), "p75": float(q[3]), "p95": float(q[4]),
        "max": vmax, "mean": mean, "std": std, "se": se,
        "ci95_low": mean - 1.96 * se if n_valid > 1 else float("nan"),
        "ci95_high": mean + 1.96 * se if n_valid > 1 else float("nan"),
        "range": vmax - vmin, "cv": (100.0 * std / mean) if n_valid > 1 and mean != 0 else float("nan")
    }

def _quicklook_png(src_tif: str, dst_png: str):
    """Genera PNG RGBA y devuelve bounds [[s,w],[n,e]] en EPSG:4326."""
    with rasterio.open(src_tif) as src:
        arr = src.read(1, masked=True)
        bounds = src.bounds  # left, bottom, right, top (w, s, e, n)
    data = np.array(arr, dtype=np.float64)
    mask = ~np.isfinite(data)
    if np.all(mask):
        gray = np.zeros_like(data, dtype=np.uint8)
    else:
        vmin = np.nanmin(data[~mask]); vmax = np.nanmax(data[~mask])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            gray = np.full_like(data, 127, dtype=np.uint8)
        else:
            scaled = (data - vmin) / (vmax - vmin)
            gray = np.clip(np.round(scaled * 255), 0, 255).astype(np.uint8)
    alpha = np.where(mask, 0, 255).astype(np.uint8)
    rgba = np.stack([gray, gray, gray, alpha], axis=0)  # (4, H, W)
    h, w = gray.shape
    profile = {"driver": "PNG", "height": h, "width": w, "count": 4, "dtype": "uint8"}
    with rasterio.open(dst_png, "w", **profile) as dst:
        dst.write(rgba)
    return [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]

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
    case_field: str = Form("cases"),
):
    if sep.lower() in ("\\t", "tab", "tabs"): sep = "\t"

    tmp_csv = os.path.join(TMP_DIR, csv_file.filename)
    with open(tmp_csv, "wb") as f:
        f.write(await csv_file.read())

    run_id = uuid.uuid4().hex[:8]
    out_dir = os.path.join(DATA_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)

    res = run_frontwave(
        csv_path=tmp_csv,
        out_folder=out_dir,
        lon_field=lon_field, lat_field=lat_field, date_field=date_field,
        id_field=id_field, weight_field=weight_field, case_field=case_field,
        grid_cell_m=float(grid), krige_cell_m=float(cell), contour_interval=float(contour),
        sep=sep, dayfirst=True
    )

    # GeoJSON puntos
    selected_geojson = None
    try:
        if gpd and res.get("selected_points") and os.path.exists(res["selected_points"]):
            gdf = gpd.read_file(res["selected_points"], layer="selected_pts")
            selected_geojson = os.path.join(out_dir, "selected_points.geojson")
            gdf.to_file(selected_geojson, driver="GeoJSON")
    except Exception:
        selected_geojson = None

    # Quicklooks PNG + bounds para Leaflet (sin libs extra)
    images = {}
    def add_img(key, tif_path):
        if not tif_path or not os.path.exists(tif_path): return
        png_path = os.path.join(out_dir, f"{key}.png")
        bounds = _quicklook_png(tif_path, png_path)
        rel = os.path.relpath(png_path, DATA_DIR).replace("\\", "/")
        images[key] = {"url": f"/data/{rel}", "bounds": bounds}

    add_img("kriging", res.get("kriging"))
    add_img("slope",   res.get("slope"))
    add_img("velocity",res.get("velocity"))

    def to_url(p):
        if not p: return None
        p = os.path.abspath(p).replace("\\", "/")
        rel = os.path.relpath(p, DATA_DIR).replace("\\", "/")
        return f"/data/{rel}"

    urls = {
        "kriging": to_url(res.get("kriging")),
        "slope": to_url(res.get("slope")),
        "velocity": to_url(res.get("velocity")),
        "contours": to_url(res.get("contours")),
        "ellipse": to_url(res.get("ellipse")),
        "selected_points": to_url(res.get("selected_points")),
        "selected_points_geojson": to_url(selected_geojson),
        "grid": to_url(res.get("grid")),
    }

    stats = {"velocity": _raster_stats(res["velocity"])} if res.get("velocity") else {"velocity": {"count": 0, "nodata_count": None}}

    return JSONResponse(content={"run_id": run_id, "urls": urls, "images": images, "stats": stats})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)

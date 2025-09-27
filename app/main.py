# main.py
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

# -------------------------------------------------
# Rutas absolutas para despliegue en Railway
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))      # un nivel arriba de /app
STATIC_DIR = os.path.join(ROOT_DIR, "static")                # static es hermana de app
DATA_DIR = os.path.join(BASE_DIR, "data")
TMP_DIR = os.path.join(BASE_DIR, "tmp")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# -------------------------------------------------
# Import robusto de frontwave
# -------------------------------------------------
try:
    from .frontwave import run_frontwave  # si se ejecuta como paquete (uvicorn app.main:app)
except ImportError:
    import sys
    if BASE_DIR not in sys.path:
        sys.path.append(BASE_DIR)
    from frontwave import run_frontwave  # si se ejecuta como script (uvicorn main:app)

app = FastAPI(title="FrontWave API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar directorios estáticos
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok")

@app.get("/")
def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return PlainTextResponse(
            f"index.html not found at {index_path}", status_code=500
        )
    return FileResponse(index_path, media_type="text/html")


# -------------------------------------------------
# Estadísticos de un raster
# -------------------------------------------------
def _raster_stats(path: str) -> dict:
    with rasterio.open(path) as src:
        band = src.read(1, masked=True)
    data = np.ma.compressed(band)

    n_valid = int(data.size)
    n_total = int(band.size)
    n_nodata = int(n_total - n_valid)

    if n_valid == 0:
        return {
            "count": 0, "nodata_count": n_nodata,
            "min": None, "p05": None, "p25": None, "p50": None, "p75": None, "p95": None, "max": None,
            "mean": None, "std": None, "se": None, "ci95_low": None, "ci95_high": None,
            "range": None, "cv": None
        }

    q = np.percentile(data, [5, 25, 50, 75, 95])
    vmin = float(np.min(data))
    vmax = float(np.max(data))
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1)) if n_valid > 1 else float("nan")
    se = (std / math.sqrt(n_valid)) if n_valid > 1 else float("nan")
    ci95_low = mean - 1.96 * se if n_valid > 1 else float("nan")
    ci95_high = mean + 1.96 * se if n_valid > 1 else float("nan")
    rng = vmax - vmin
    cv = (100.0 * std / mean) if n_valid > 1 and mean != 0 else float("nan")

    return {
        "count": n_valid,
        "nodata_count": n_nodata,
        "min": vmin,
        "p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]), "p75": float(q[3]), "p95": float(q[4]),
        "max": vmax,
        "mean": mean,
        "std": std,
        "se": se,
        "ci95_low": ci95_low, "ci95_high": ci95_high,
        "range": rng, "cv": cv
    }


# -------------------------------------------------
# Endpoint principal de proceso
# -------------------------------------------------
@app.post("/run")
async def run_process(
    csv_file: UploadFile,
    grid: int = Form(...),
    cell: int = Form(...),
    contour: int = Form(...),
):
    # guardar CSV temporal
    tmp_csv = os.path.join(TMP_DIR, csv_file.filename)
    with open(tmp_csv, "wb") as f:
        f.write(await csv_file.read())

    run_id = uuid.uuid4().hex[:8]
    out_dir = os.path.join(DATA_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)

    res = run_frontwave(
        csv_path=tmp_csv,
        out_folder=out_dir,
        grid_cell_m=float(grid),
        krige_cell_m=float(cell),
        contour_interval=float(contour),
        sep=';',
        dayfirst=True
    )

    def to_url(p):
        if not p:
            return None
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
        "grid": to_url(res.get("grid")),
    }

    stats = {}
    if res.get("velocity"):
        stats["velocity"] = _raster_stats(res["velocity"])
    else:
        stats["velocity"] = {"count": 0, "nodata_count": None}

    return JSONResponse(content={"run_id": run_id, "urls": urls, "stats": stats})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    # si ejecutas como paquete (uvicorn app.main:app) no hace falta este __main__,
    # pero no afecta si lo dejas
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)

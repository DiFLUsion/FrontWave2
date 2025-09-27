# main.py
import os
import math
import uuid
import numpy as np
import rasterio
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from frontwave import run_frontwave  # módulo que genera los resultados

app = FastAPI(title="FrontWave API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir estáticos y salidas
os.makedirs("static", exist_ok=True)
os.makedirs("data", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/data", StaticFiles(directory="data"), name="data")

@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ---------------------------------------------
# Estadísticos del raster (celdas válidas)
# ---------------------------------------------
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


# ---------------------------------------------
# Endpoint principal
# ---------------------------------------------
@app.post("/run")
async def run_process(
    csv_file: UploadFile,
    grid: int = Form(...),      # tamaño de celda de selección temprana (m)
    cell: int = Form(...),      # tamaño de celda del kriging (m)
    contour: int = Form(...),   # intervalo de isolíneas
):
    # Guardar CSV temporalmente
    os.makedirs("tmp", exist_ok=True)
    tmp_csv = os.path.join("tmp", csv_file.filename)
    with open(tmp_csv, "wb") as f:
        f.write(await csv_file.read())

    # Carpeta de salida única por ejecución
    run_id = uuid.uuid4().hex[:8]
    out_dir = os.path.join("data", run_id)
    os.makedirs(out_dir, exist_ok=True)

    # Ejecutar proceso principal
    res = run_frontwave(
        csv_path=tmp_csv,
        out_folder=out_dir,
        grid_cell_m=float(grid),
        krige_cell_m=float(cell),
        contour_interval=float(contour),
        sep=';',
        dayfirst=True
    )

    # Convertir rutas locales a URLs servidas en /data
    def to_url(p):
        if not p:
            return None
        p = p.replace("\\", "/")
        # asegúrate de que está bajo data/
        rel = p.split("data/", 1)[-1] if "data/" in p else os.path.relpath(p, "data").replace("\\", "/")
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

    # Estadísticas
    stats = {}
    if res.get("velocity"):
        stats["velocity"] = _raster_stats(res["velocity"])
    else:
        stats["velocity"] = {"count": 0, "nodata_count": None}

    payload = {"run_id": run_id, "urls": urls, "stats": stats}
    return JSONResponse(content=payload)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

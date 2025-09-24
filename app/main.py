# app/main.py
import os
import math
import numpy as np
import rasterio
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.frontwave import run_frontwave # tu módulo que genera los resultados
 
app = FastAPI(title="FrontWave API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------
# Función para estadísticas del raster (full raster, excluyendo NoData)
# -------------------------------------------------------------------
def _raster_stats(path: str) -> dict:
    """
    Estadísticos del raster completo sobre celdas válidas (excluye NoData).
    Devuelve: count, nodata_count, min, p05, p25, p50, p75, p95, max,
              mean, std, se, ci95_low, ci95_high, range, cv
    """
    with rasterio.open(path) as src:
        band = src.read(1, masked=True)  # MaskedArray con NoData ya en mask
    data = np.ma.compressed(band)        # solo válidos

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


# -------------------------------------------------------------------
# Endpoint principal
# -------------------------------------------------------------------
@app.post("/run")
async def run_process(
    csv_file: UploadFile,
    grid: int = Form(...),
    cell: int = Form(...),
    contour: int = Form(...),
):
    """
    Lanza el proceso FrontWave y devuelve URLs y estadísticas.
    """
    # Guardar CSV temporalmente
    tmp_csv = os.path.join("tmp", csv_file.filename)
    os.makedirs("tmp", exist_ok=True)
    with open(tmp_csv, "wb") as f:
        f.write(await csv_file.read())

    # Ejecutar proceso principal (frontwave.py)
    res = run_frontwave(tmp_csv, grid=grid, cell=cell, contour=contour)

    # URLs a devolver
    urls = {
        "velocity": res.get("velocity_url"),
        "kriging": res.get("kriging_url"),
        "contour": res.get("contour_url"),
        "slope": res.get("slope_url"),
    }

    # Estadísticas
    stats = {}

    # Velocity: usar la función nueva
    if res.get("velocity"):
        stats["velocity"] = _raster_stats(res["velocity"])
    else:
        stats["velocity"] = {"count": 0, "nodata_count": None}

    # Mantén si quieres el min/max para otros
    if res.get("kriging_minmax"):
        stats["kriging"] = res["kriging_minmax"]
    if res.get("slope_minmax"):
        stats["slope"] = res["slope_minmax"]

    payload = {"urls": urls, "stats": stats}
    return JSONResponse(content=payload)


# -------------------------------------------------------------------
# Main local
# -------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# app/frontwave.py
# -*- coding: utf-8 -*-
import os, math, warnings, sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
from pykrige.ok import OrdinaryKriging
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS as RCRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from skimage import measure

CONDA_PREFIX = os.environ.get("CONDA_PREFIX", sys.prefix)
proj_dir = os.path.join(CONDA_PREFIX, "Library", "share", "proj")
gdal_dir = os.path.join(CONDA_PREFIX, "Library", "share", "gdal")
bin_dir  = os.path.join(CONDA_PREFIX, "Library", "bin")

os.environ.setdefault("PROJ_LIB", proj_dir)
os.environ.setdefault("GDAL_DATA", gdal_dir)
os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH","")

try:
    from pyproj import datadir as _pdd
    os.environ.setdefault("PROJ_LIB", _pdd.get_data_dir())
except Exception:
    pass

def guess_utm_epsg(lon, lat):
    zone = int((lon + 180) / 6) + 1
    south = lat < 0
    return f"EPSG:{32700 + zone if south else 32600 + zone}"

def to_metric_crs(gdf):
    cx, cy = gdf.geometry.unary_union.centroid.xy
    epsg = guess_utm_epsg(cx[0], cy[0])
    return gdf.to_crs(epsg), epsg

def _pick_column(df, candidates, required=True):
    m = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        k = c.lower().strip()
        if k in m:
            return m[k]
    if required:
        raise ValueError(f"Missing columns {candidates}. Available: {list(df.columns)}")
    return None

def read_points_from_csv(csv_path, lon_field="lon", lat_field="lat",
                         date_field="date", id_field="id",
                         weight_field="weight", case_field="cases",
                         sep=';', date_format=None, dayfirst=True):
    df = pd.read_csv(csv_path, sep=sep)

    lon_field  = _pick_column(df, [lon_field, "lon","long","longitude","x","lon_wgs84"])
    lat_field  = _pick_column(df, [lat_field, "lat","latitude","y","lat_wgs84"])
    date_field = _pick_column(df, [date_field, "fecha","date","datetime","fecha_hora","fec"], required=True)
    id_field   = _pick_column(df, [id_field, "id","identificador","codigo"], required=False)
    weight_field = _pick_column(df, [weight_field, "peso","w","weight"], required=False)
    case_field = _pick_column(df, [case_field, "casos","n","count","cases"], required=False)

    ref = pd.Timestamp("1901-01-01")
    if date_format:
        dt = pd.to_datetime(df[date_field], format=date_format, errors="coerce")
    else:
        try:
            dt = pd.to_datetime(df[date_field], format="mixed", dayfirst=dayfirst, errors="coerce")
        except TypeError:
            dt = pd.to_datetime(df[date_field], dayfirst=dayfirst, errors="coerce")
    if dt.isna().all():
        raise ValueError(f"Cannot parse dates in '{date_field}'. Set 'date_format' or 'dayfirst'.")

    df = df.loc[~dt.isna()].copy()
    df["N1"] = (dt.loc[~dt.isna()] - ref).dt.days.astype(int)
    df["DATE_ISO"] = pd.to_datetime(df[date_field], errors="coerce").dt.strftime("%Y-%m-%d")
    df["DATE_ISO"] = df["DATE_ISO"].fillna("")

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_field], df[lat_field]), crs="EPSG:4326")
    meta = {"date": date_field, "id": id_field, "weight": weight_field, "case": case_field}
    return gdf, meta

def make_square_grid(bounds, cell_size):
    xmin, ymin, xmax, ymax = bounds
    cols = int(math.ceil((xmax - xmin) / cell_size))
    rows = int(math.ceil((ymax - ymin) / cell_size))
    polys = []
    for i in range(cols):
        for j in range(rows):
            x0 = xmin + i * cell_size
            y0 = ymin + j * cell_size
            polys.append(Polygon([(x0, y0),(x0+cell_size, y0),(x0+cell_size, y0+cell_size),(x0, y0+cell_size)]))
    return gpd.GeoDataFrame(geometry=polys, crs=None)

def select_earliest_points_by_cell(points_m, cell_size):
    grid = make_square_grid(points_m.total_bounds, cell_size).set_crs(points_m.crs)
    join = gpd.sjoin(points_m, grid.reset_index().rename(columns={"index": "GRID_ID"}), how="left", predicate="within")
    min_by_cell = join.groupby("GRID_ID")["N1"].min().rename("MIN_N1").reset_index()
    join2 = join.merge(min_by_cell, on="GRID_ID", how="left")
    selected = join2[join2["N1"] == join2["MIN_N1"]].copy()
    min_global = selected["MIN_N1"].min()
    selected["NUM0"] = selected["MIN_N1"] - min_global
    return selected, grid

def run_ok_kriging(selected_m, value_field="NUM0", cell_size=1000.0, variogram_model="spherical"):
    xs = selected_m.geometry.x.values
    ys = selected_m.geometry.y.values
    zs = selected_m[value_field].values.astype(float)
    xmin, ymin, xmax, ymax = selected_m.total_bounds
    nx = int(math.ceil((xmax - xmin) / cell_size)) + 1
    ny = int(math.ceil((ymax - ymin) / cell_size)) + 1
    gridx = np.linspace(xmin, xmin + cell_size * (nx - 1), nx)
    gridy = np.linspace(ymin, ymin + cell_size * (ny - 1), ny)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        OK = OrdinaryKriging(xs, ys, zs, variogram_model=variogram_model, enable_plotting=False, verbose=False)
        zgrid, _ = OK.execute("grid", gridx, gridy)
    return gridx, gridy, np.asarray(zgrid)

def save_geotiff_utm(path, gridx, gridy, z, epsg_code):
    z_rev = np.flipud(z)
    resx = gridx[1] - gridx[0] if len(gridx) > 1 else 1.0
    resy = gridy[1] - gridy[0] if len(gridy) > 1 else 1.0
    transform = from_origin(gridx.min(), gridy.max(), resx, resy)
    crs = RCRS.from_string(epsg_code)
    profile = {
        "driver": "GTiff", "height": z_rev.shape[0], "width": z_rev.shape[1],
        "count": 1, "dtype": z_rev.dtype, "crs": crs, "transform": transform,
        "compress": "deflate", "tiled": True, "nodata": np.nan
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(z_rev, 1)

def reproject_to_wgs84(src_path, dst_path):
    dst_crs = "EPSG:4326"
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        profile = src.profile.copy()
        profile.update({"crs": dst_crs, "transform": transform, "width": width, "height": height})
        with rasterio.open(dst_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=transform, dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
    return dst_path

def contours_from_grid(gridx, gridy, z, interval):
    dx = gridx[1] - gridx[0]
    dy = gridy[1] - gridy[0]
    x0 = gridx.min()
    y0 = gridy.min()
    zmin, zmax = np.nanmin(z), np.nanmax(z)
    levels = np.arange(np.floor(zmin/interval)*interval, np.ceil(zmax/interval)*interval + interval, interval)
    lines = []
    for lev in levels:
        cs = measure.find_contours(z, lev)
        for arr in cs:
            yy = y0 + arr[:, 0] * dy
            xx = x0 + arr[:, 1] * dx
            if len(xx) >= 2:
                lines.append({"level": lev, "geometry": LineString(np.c_[xx, yy])})
    return gpd.GeoDataFrame(lines, geometry="geometry")

def slope_degrees_from_grid(z, cell_size):
    dz_dy, dz_dx = np.gradient(z, cell_size, cell_size)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    return np.degrees(slope_rad)

def velocity_from_slope(slope_deg):
    vel = np.full_like(slope_deg, np.nan, dtype=float)
    mask = slope_deg > 0
    vel[mask] = 1.0 / slope_deg[mask]
    return vel

def standard_deviational_ellipse(points_m, weight_field=None, group_field=None):
    df = points_m.copy()
    df["_w"] = 1.0 if weight_field is None else df[weight_field].astype(float).fillna(0.0)
    def ellipse_for_group(gg):
        x = gg.geometry.x.values; y = gg.geometry.y.values; w = gg["_w"].values
        W = w.sum()
        if W == 0 or len(x) < 2: return None
        mx = np.average(x, weights=w); my = np.average(y, weights=w)
        x0, y0 = x - mx, y - my
        Sxx = np.average(x0*x0, weights=w); Syy = np.average(y0*y0, weights=w); Sxy = np.average(x0*y0, weights=w)
        cov = np.array([[Sxx, Sxy],[Sxy, Syy]])
        vals, vecs = np.linalg.eigh(cov)
        a = math.sqrt(vals[1]); b = math.sqrt(vals[0])
        theta = math.atan2(vecs[1,1], vecs[0,1])
        t = np.linspace(0, 2*np.pi, 256)
        ex, ey = a*np.cos(t), b*np.sin(t)
        R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        e = R @ np.vstack([ex, ey])
        xx = e[0,:] + mx; yy = e[1,:] + my
        return Polygon(np.c_[xx, yy])
    if group_field:
        out = []
        for key, gg in df.groupby(group_field):
            poly = ellipse_for_group(gg)
            if poly: out.append({group_field: key, "geometry": poly})
        return gpd.GeoDataFrame(out, geometry="geometry", crs=df.crs)
    poly = ellipse_for_group(df)
    return gpd.GeoDataFrame([{"geometry": poly}], geometry="geometry", crs=df.crs) if poly else gpd.GeoDataFrame(geometry=[], crs=df.crs)

def run_frontwave(csv_path, out_folder,
                  lon_field="lon", lat_field="lat", date_field="date", id_field="id",
                  weight_field="weight", case_field="cases",
                  grid_cell_m=12000.0, krige_cell_m=1200.0, contour_interval=30.0,
                  sep=';', date_format=None, dayfirst=True):
    os.makedirs(out_folder, exist_ok=True)

    # ALL points WGS84 with DATE_ISO
    pts_wgs84, meta = read_points_from_csv(
        csv_path, lon_field, lat_field, date_field, id_field,
        weight_field, case_field, sep=sep,
        date_format=date_format, dayfirst=dayfirst
    )

    # Save ALL points to GPKG and GeoJSON (WGS84)
    all_points_gpkg = os.path.join(out_folder, "all_points.gpkg")
    pts_wgs84.to_file(all_points_gpkg, layer="all_pts", driver="GPKG")
    all_points_geojson = os.path.join(out_folder, "all_points.geojson")
    pts_wgs84.to_file(all_points_geojson, driver="GeoJSON")

    # Project for analysis
    pts_m, epsg = to_metric_crs(pts_wgs84)
    selected_m, grid = select_earliest_points_by_cell(pts_m, grid_cell_m)

    # Add DATE_ISO to selected subset and export both GPKG and GeoJSON (WGS84)
    dcol = meta["date"]
    try:
        dt_sel = pd.to_datetime(selected_m[dcol], errors="coerce")
        selected_m["DATE_ISO"] = dt_sel.dt.strftime("%Y-%m-%d").fillna("")
    except Exception:
        selected_m["DATE_ISO"] = ""

    selected_gpkg = os.path.join(out_folder, "selected_points.gpkg")
    selected_m.to_file(selected_gpkg, layer="selected_pts", driver="GPKG")
    selected_points_geojson = os.path.join(out_folder, "selected_points.geojson")
    selected_m.to_crs("EPSG:4326").to_file(selected_points_geojson, driver="GeoJSON")

    # Kriging and rasters
    gx, gy, z = run_ok_kriging(selected_m, value_field="NUM0", cell_size=krige_cell_m, variogram_model="spherical")

    kriging_utm = os.path.join(out_folder, "kriging_utm.tif")
    save_geotiff_utm(kriging_utm, gx, gy, z, epsg)
    kriging_tif = os.path.join(out_folder, "kriging.tif")
    reproject_to_wgs84(kriging_utm, kriging_tif)

    slope = slope_degrees_from_grid(z, krige_cell_m)
    slope_utm = os.path.join(out_folder, "slope_utm.tif")
    save_geotiff_utm(slope_utm, gx, gy, slope.astype(np.float32), epsg)
    slope_tif = os.path.join(out_folder, "slope.tif")
    reproject_to_wgs84(slope_utm, slope_tif)

    velocity = velocity_from_slope(slope)
    velocity_utm = os.path.join(out_folder, "velocity_utm.tif")
    save_geotiff_utm(velocity_utm, gx, gy, velocity.astype(np.float32), epsg)
    velocity_tif = os.path.join(out_folder, "velocity.tif")
    reproject_to_wgs84(velocity_utm, velocity_tif)

    # Vectors: contours and ellipse → GPKG + GeoJSON (WGS84)
    gcont = contours_from_grid(gx, gy, z, interval=contour_interval).set_crs(epsg)
    contours_gpkg = os.path.join(out_folder, "contours.gpkg")
    if len(gcont): gcont.to_file(contours_gpkg, layer="contours", driver="GPKG")
    contours_geojson = os.path.join(out_folder, "contours.geojson")
    if len(gcont): gcont.to_crs("EPSG:4326").to_file(contours_geojson, driver="GeoJSON")

    ellipse = standard_deviational_ellipse(pts_m, weight_field=meta["weight"], group_field=None)
    ellipse_gpkg = os.path.join(out_folder, "ellipse.gpkg")
    if len(ellipse): ellipse.to_file(ellipse_gpkg, layer="ellipse", driver="GPKG")
    ellipse_geojson = os.path.join(out_folder, "ellipse.geojson")
    if len(ellipse): ellipse.to_crs("EPSG:4326").to_file(ellipse_geojson, driver="GeoJSON")

    grid_gpkg = os.path.join(out_folder, "grid.gpkg")
    grid.to_file(grid_gpkg, layer="grid", driver="GPKG")

    return {
        "kriging": kriging_tif,
        "contours": contours_gpkg,
        "contours_geojson": contours_geojson if os.path.exists(contours_geojson) else None,
        "slope": slope_tif,
        "velocity": velocity_tif,
        "ellipse": ellipse_gpkg,
        "ellipse_geojson": ellipse_geojson if os.path.exists(ellipse_geojson) else None,
        "selected_points": selected_gpkg,
        "selected_points_geojson": selected_points_geojson if os.path.exists(selected_points_geojson) else None,
        "all_points": all_points_gpkg,
        "all_points_geojson": all_points_geojson if os.path.exists(all_points_geojson) else None,
        "grid": grid_gpkg,
        "crs": epsg,
        "date_field": dcol
    }

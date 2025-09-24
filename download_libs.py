import os, requests

libs = {
    "leaflet.css": "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css",
    "leaflet.js":  "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js",
    "geotiff.min.js": "https://cdn.jsdelivr.net//npm/geotiff@2.1.4-beta.0/dist-browser/geotiff.js",
    "plotty.min.js":  "https://cdn.jsdelivr.net/npm/plotty@0.4.0/dist/plotty.min.js",
    "leaflet-geotiff.min.js": "https://cdn.jsdelivr.net//npm/leaflet-geotiff-2@1.1.0/dist/leaflet-geotiff.js",
    "leaflet-geotiff-plotty.min.js": "https://cdn.jsdelivr.net/npm/leaflet-geotiff-2@1.1.0/dist/leaflet-geotiff-plotty.min.js",
}

outdir = os.path.join("static", "libs")
os.makedirs(outdir, exist_ok=True)

for fname, url in libs.items():
    print("Descargando", fname, "...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(os.path.join(outdir, fname), "wb") as f:
        f.write(r.content)

print("OK ->", outdir)

import ee

def process_water_boundaries(bbox, threshold=0.1, area_min=8000):

    roi = ee.Geometry.Rectangle(bbox)

    # Sentinel-2
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate("2023-11-01", "2024-03-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 5))
        .map(lambda img: img.updateMask(
            img.select("SCL")
               .neq(3).And(img.select("SCL").neq(8))
               .And(img.select("SCL").neq(9))
               .And(img.select("SCL").neq(10))
               .And(img.select("SCL").neq(11))
        ))
        .median()
        .clip(roi)
    )

    ndwi  = s2.normalizedDifference(["B3", "B8"]).rename("NDWI")
    mndwi = s2.normalizedDifference(["B3", "B11"]).rename("MNDWI")

    water_mask = (
        ndwi.gt(threshold).Or(mndwi.gt(0.0))
            .And(ndwi.gt(-0.2))
            .And(mndwi.gt(-0.2))
    )

    cleaned = (
        water_mask
        .focal_min(radius=1, kernelType='circle', iterations=1)
        .focal_max(radius=2, kernelType='circle', iterations=1)
    )
    connected = cleaned.connectedPixelCount(50, False)
    cleaned   = cleaned.updateMask(connected.gte(15))

    vectors = cleaned.selfMask().reduceToVectors(
        geometry=roi,
        scale=20,
        geometryType="polygon",
        labelProperty="water",
        maxPixels=int(1e9),
        bestEffort=True,
        tileScale=4
    )
    vectors = vectors.map(lambda f: f.set("area_m2", f.geometry().area(1)))
    vectors = vectors.filter(ee.Filter.gt("area_m2", area_min))

    # ── Tile URLs ─────────────────────────────────────────────────
    s2_map = s2.visualize(
        bands=["B4", "B3", "B2"], min=0, max=3000, gamma=1.2
    ).getMapId()

    ndwi_map = ndwi.visualize(
        min=-0.3, max=0.5,
        palette=["#8B4513","#DAA520","#228B22","#00CED1","#0000FF"]
    ).getMapId()

    mask_map = cleaned.selfMask().visualize(
        palette=["#00BFFF"]
    ).getMapId()

    def tile_url(map_id_obj):
        return (
            "https://earthengine.googleapis.com/v1/"
            + map_id_obj["mapid"]
            + "/tiles/{z}/{x}/{y}"
        )

    # ── Safe GeoJSON build ────────────────────────────────────────
    raw = vectors.getInfo()

    # Normalize whatever GEE returns into valid FeatureCollection
    if isinstance(raw, dict) and raw.get("type") == "FeatureCollection":
        geojson = raw
    elif isinstance(raw, dict) and "features" in raw:
        geojson = {"type": "FeatureCollection", "features": raw["features"]}
    elif isinstance(raw, list):
        geojson = {"type": "FeatureCollection", "features": raw}
    else:
        geojson = {"type": "FeatureCollection", "features": []}

    # ── Clean each feature — Leaflet needs exact structure ────────
    clean_features = []
    for f in geojson.get("features", []):
        # Skip anything without valid geometry
        if not f or not isinstance(f, dict):
            continue
        geom = f.get("geometry")
        if not geom or not geom.get("type") or not geom.get("coordinates"):
            continue
        clean_features.append({
            "type":       "Feature",
            "geometry":   geom,
            "properties": f.get("properties", {})
        })

    final_geojson = {
        "type":     "FeatureCollection",
        "features": clean_features
    }

    return {
        "geojson":       final_geojson,
        "feature_count": len(clean_features),
        "tile_urls": {
            "satellite":  tile_url(s2_map),
            "ndwi":       tile_url(ndwi_map),
            "water_mask": tile_url(mask_map),
        }
    }
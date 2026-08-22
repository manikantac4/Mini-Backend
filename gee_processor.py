import ee


def process_water_boundaries(
    latitude,
    longitude,
    radius_km=10,
    threshold=0.1,
    area_min=8000
):

    # ============================================================
    # 1. CREATE STUDY AREA USING RADIUS
    # ============================================================

    center = ee.Geometry.Point([
        float(longitude),
        float(latitude)
    ])

    # Earth Engine buffer uses metres
    roi = center.buffer(float(radius_km) * 1000)

    # ============================================================
    # 2. LOAD SENTINEL-2
    # ============================================================

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate("2023-11-01", "2024-03-31")
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                20
            )
        )
    )

    # ============================================================
    # 3. CLOUD / SHADOW MASK
    # ============================================================

    def mask_s2(image):

        scl = image.select("SCL")

        mask = (
            scl.neq(3)
            .And(scl.neq(8))
            .And(scl.neq(9))
            .And(scl.neq(10))
            .And(scl.neq(11))
        )

        return image.updateMask(mask)

    s2 = (
        collection
        .map(mask_s2)
        .median()
        .clip(roi)
    )

    # ============================================================
    # 4. WATER INDICES
    # ============================================================

    ndwi = (
        s2.normalizedDifference(["B3", "B8"])
        .rename("NDWI")
    )

    mndwi = (
        s2.normalizedDifference(["B3", "B11"])
        .rename("MNDWI")
    )

    # ============================================================
    # 5. CREATE WATER MASK
    # ============================================================

    water_mask = (
        ndwi.gt(float(threshold))
        .Or(mndwi.gt(0.0))
        .And(ndwi.gt(-0.2))
        .And(mndwi.gt(-0.2))
    )

    # ============================================================
    # 6. CLEAN WATER MASK
    # ============================================================

    cleaned = (
        water_mask
        .focal_min(
            radius=1,
            kernelType="circle",
            iterations=1
        )
        .focal_max(
            radius=2,
            kernelType="circle",
            iterations=1
        )
    )

    connected = cleaned.connectedPixelCount(
        50,
        False
    )

    cleaned = cleaned.updateMask(
        connected.gte(15)
    )

    # ============================================================
    # 7. CONVERT WATER TO POLYGONS
    # ============================================================

    vectors = cleaned.selfMask().reduceToVectors(
        geometry=roi,
        scale=20,
        geometryType="polygon",
        labelProperty="water",
        maxPixels=int(1e9),
        bestEffort=True,
        tileScale=4
    )

    # ============================================================
    # 8. CALCULATE AREA
    # ============================================================

    def add_area(feature):

        area = feature.geometry().area(1)

        return feature.set({
            "area_m2": area,
            "area_km2": area.divide(1_000_000)
        })

    vectors = vectors.map(add_area)

    # Remove tiny detections
    vectors = vectors.filter(
        ee.Filter.gt(
            "area_m2",
            float(area_min)
        )
    )

    # ============================================================
    # 9. SATELLITE VISUALIZATION
    # ============================================================

    s2_map = s2.visualize(
        bands=["B4", "B3", "B2"],
        min=0,
        max=3000,
        gamma=1.2
    ).getMapId()

    # ============================================================
    # 10. NDWI VISUALIZATION
    # ============================================================

    ndwi_map = ndwi.visualize(
        min=-0.3,
        max=0.5,
        palette=[
            "#8B4513",
            "#DAA520",
            "#228B22",
            "#00CED1",
            "#0000FF"
        ]
    ).getMapId()

    # ============================================================
    # 11. WATER MASK VISUALIZATION
    # ============================================================

    mask_map = (
        cleaned
        .selfMask()
        .visualize(
            palette=["#00BFFF"]
        )
        .getMapId()
    )

    # ============================================================
    # 12. TILE URL FUNCTION
    # ============================================================

    def tile_url(map_object):

        return (
            "https://earthengine.googleapis.com/v1/"
            + map_object["mapid"]
            + "/tiles/{z}/{x}/{y}"
        )

    # ============================================================
    # 13. GET GEOJSON
    # ============================================================

    raw = vectors.getInfo()

    if (
        isinstance(raw, dict)
        and raw.get("type") == "FeatureCollection"
    ):

        features = raw.get(
            "features",
            []
        )

    elif (
        isinstance(raw, dict)
        and "features" in raw
    ):

        features = raw["features"]

    elif isinstance(raw, list):

        features = raw

    else:

        features = []

    # ============================================================
    # 14. CLEAN + NUMBER WATER BODIES
    # ============================================================

    clean_features = []

    # IMPORTANT:
    # numbering is based only on VALID features,
    # so we always get 1,2,3,4... without gaps.

    for feature in features:

        if not isinstance(feature, dict):
            continue

        geometry = feature.get(
            "geometry"
        )

        if not geometry:
            continue

        if not geometry.get("type"):
            continue

        if not geometry.get("coordinates"):
            continue

        properties = feature.get(
            "properties",
            {}
        ).copy()

        # --------------------------------------------
        # Sequential Water Body ID
        # --------------------------------------------

        water_body_id = (
            len(clean_features) + 1
        )

        properties["water_body_id"] = (
            water_body_id
        )

        properties["water_body_name"] = (
            f"Water Body {water_body_id}"
        )

        # --------------------------------------------
        # Ensure area values exist
        # --------------------------------------------

        try:

            area_m2 = float(
                properties.get(
                    "area_m2",
                    0
                )
            )

        except (TypeError, ValueError):

            area_m2 = 0

        properties["area_m2"] = area_m2

        properties["area_km2"] = (
            area_m2 / 1_000_000
        )

        clean_features.append({

            "type": "Feature",

            "geometry": geometry,

            "properties": properties

        })

    # ============================================================
    # 15. FINAL GEOJSON
    # ============================================================

    final_geojson = {

        "type": "FeatureCollection",

        "features": clean_features

    }

    # ============================================================
    # 16. GET ROI BOUNDS
    # ============================================================

    roi_bounds = (
        roi
        .bounds()
        .coordinates()
        .getInfo()[0]
    )

    # Convert Earth Engine polygon coordinates into
    # west/south/east/north values

    longitudes = [
        coordinate[0]
        for coordinate in roi_bounds
    ]

    latitudes = [
        coordinate[1]
        for coordinate in roi_bounds
    ]

    bbox = [
        min(longitudes),  # west
        min(latitudes),   # south
        max(longitudes),  # east
        max(latitudes)    # north
    ]

    # ============================================================
    # 17. RETURN RESULT
    # ============================================================

    return {

        "geojson":
            final_geojson,

        "feature_count":
            len(clean_features),

        "center": {

            "latitude":
                float(latitude),

            "longitude":
                float(longitude)

        },

        "radius_km":
            float(radius_km),

        "bbox":
            bbox,

        "tile_urls": {

            "satellite":
                tile_url(s2_map),

            "ndwi":
                tile_url(ndwi_map),

            "water_mask":
                tile_url(mask_map)

        }
    }
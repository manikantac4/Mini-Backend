import ee


# ================================================================
# TUNABLE PARAMETERS
#
# Centralised here so the algorithm can be tuned without touching
# the detection logic itself. Callers of process_water_boundaries()
# can override any of these per-request; anything not passed in
# falls back to the defaults below.
# ================================================================

DEFAULT_NDWI_THRESHOLD = 0.10
DEFAULT_NDWI_SECONDARY_THRESHOLD = 0.00   # used alongside AWEI
DEFAULT_MNDWI_THRESHOLD = 0.00
DEFAULT_AWEI_THRESHOLD = 0.00

DEFAULT_MAX_NDVI = 0.35   # vegetation exclusion ceiling
DEFAULT_MAX_NDBI = 0.20   # built-up exclusion ceiling

DEFAULT_MIN_CONNECTED_PIXELS = 15
DEFAULT_MIN_AREA_M2 = 8000

# Final vector boundary scale. 10 m matches the native resolution
# of the visible/NIR bands driving NDWI/NDVI; MNDWI/AWEI (built on
# the 20 m SWIR bands) are resampled by GEE when combined at this
# scale, so the boundary is sharper without pretending the SWIR
# inputs are natively 10 m.
BOUNDARY_SCALE_M = 10

# Very slight simplification so shorelines aren't left jagged by
# raster-to-vector conversion, without flattening natural lake
# shapes into geometric-looking polygons.
SIMPLIFY_TOLERANCE_M = 8

# Cloud cover filter for the Sentinel-2 collection.
MAX_CLOUDY_PIXEL_PERCENTAGE = 20

# SCL classes removed during cloud/shadow masking.
CLOUD_SHADOW_SCL_CLASSES = [3, 8, 9, 10, 11]


# ================================================================
# WATER BODY PROCESSOR
# ================================================================

def process_water_boundaries(
    latitude,
    longitude,
    radius_km=50,
    threshold=DEFAULT_NDWI_THRESHOLD,
    area_min=DEFAULT_MIN_AREA_M2,
    ndwi_secondary_threshold=DEFAULT_NDWI_SECONDARY_THRESHOLD,
    mndwi_threshold=DEFAULT_MNDWI_THRESHOLD,
    awei_threshold=DEFAULT_AWEI_THRESHOLD,
    max_ndvi=DEFAULT_MAX_NDVI,
    max_ndbi=DEFAULT_MAX_NDBI,
    min_connected_pixels=DEFAULT_MIN_CONNECTED_PIXELS,
    date_start="2023-11-01",
    date_end="2024-03-31",
):

    # ============================================================
    # 1. CENTER POINT
    # ============================================================

    latitude = float(latitude)
    longitude = float(longitude)
    radius_km = float(radius_km)

    ndwi_threshold = float(threshold)
    ndwi_secondary_threshold = float(ndwi_secondary_threshold)
    mndwi_threshold = float(mndwi_threshold)
    awei_threshold = float(awei_threshold)
    max_ndvi = float(max_ndvi)
    max_ndbi = float(max_ndbi)
    min_connected_pixels = int(min_connected_pixels)
    area_min = float(area_min)

    center = ee.Geometry.Point([
        longitude,
        latitude
    ])


    # ============================================================
    # 2. WATER DETECTION REGION
    #
    # This is the actual circular scanning region.
    # ============================================================

    roi = center.buffer(
        radius_km * 1000
    )


    # ============================================================
    # 3. DISPLAY REGION
    #
    # IMPORTANT:
    # We DO NOT clip the satellite image to the circle.
    #
    # This gives us a normal satellite map around the circle
    # instead of white space outside the detection region.
    #
    # 1.25 gives some extra map area around the scan circle.
    # ============================================================

    display_radius_km = radius_km * 1.25

    display_roi = center.buffer(
        display_radius_km * 1000
    )

    display_bbox = (
        display_roi
        .bounds()
        .coordinates()
        .getInfo()[0]
    )

    display_longitudes = [
        coordinate[0]
        for coordinate in display_bbox
    ]

    display_latitudes = [
        coordinate[1]
        for coordinate in display_bbox
    ]

    display_bounds = [
        min(display_longitudes),
        min(display_latitudes),
        max(display_longitudes),
        max(display_latitudes)
    ]


    # ============================================================
    # 4. LOAD SENTINEL-2
    #
    # Filter using the actual circular detection region.
    # Keep a consistent season (dry-season default below) so
    # comparisons across cities/runs stay meaningful.
    # ============================================================

    collection = (
        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
        )
        .filterBounds(roi)
        .filterDate(
            date_start,
            date_end
        )
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                MAX_CLOUDY_PIXEL_PERCENTAGE
            )
        )
    )


    # ============================================================
    # 5. CLOUD / SHADOW MASK
    # ============================================================

    def mask_s2(image):

        scl = image.select("SCL")

        mask = ee.Image.constant(1)

        for scl_class in CLOUD_SHADOW_SCL_CLASSES:
            mask = mask.And(scl.neq(scl_class))

        return image.updateMask(mask)


    # ============================================================
    # 6. CREATE SENTINEL COMPOSITE
    #
    # All water indices below are computed AFTER masking, so
    # clouds/shadows/cirrus/snow never leak into the composite.
    # ============================================================

    s2 = (
        collection
        .map(mask_s2)
        .median()
    )


    # ============================================================
    # 7. DETECTION IMAGE
    #
    # Only the circular ROI is used for water detection.
    # ============================================================

    s2_detection = s2.clip(
        roi
    )


    # ============================================================
    # 8. WATER / VEGETATION / BUILT-UP INDICES
    # ============================================================

    # NDWI (McFeeters) — primary open-water signal.
    ndwi = (
        s2_detection
        .normalizedDifference(["B3", "B8"])
        .rename("NDWI")
    )

    # MNDWI (Xu) — separates water from urban/built-up surfaces
    # much better than NDWI alone.
    mndwi = (
        s2_detection
        .normalizedDifference(["B3", "B11"])
        .rename("MNDWI")
    )

    # AWEI (Automated Water Extraction Index) — independent water
    # signal that is particularly useful for shadow rejection and
    # dark-surface discrimination.
    awei = (
        s2_detection.select("B3").multiply(4)
        .subtract(
            s2_detection.select("B11").multiply(4)
        )
        .subtract(
            s2_detection.select("B8").multiply(0.25)
        )
        .subtract(
            s2_detection.select("B12").multiply(2.75)
        )
        .rename("AWEI")
    )

    # NDVI — used to exclude vegetation, which can otherwise
    # produce a high-NIR response that mimics water.
    ndvi = (
        s2_detection
        .normalizedDifference(["B8", "B4"])
        .rename("NDVI")
    )

    # NDBI — used to suppress false detections over built-up /
    # urban surfaces.
    ndbi = (
        s2_detection
        .normalizedDifference(["B11", "B8"])
        .rename("NDBI")
    )

    vegetation_free = ndvi.lt(max_ndvi)
    builtup_free = ndbi.lt(max_ndbi)


    # ============================================================
    # 9. MULTI-INDEX WATER DECISION
    #
    # A pixel is only classified as water when NDWI and MNDWI
    # agree, OR when AWEI confirms a secondary NDWI signal. This
    # is stricter than "any single index positive" and removes a
    # large share of false positives. Vegetation and built-up
    # exclusion masks are then applied on top.
    # ============================================================

    water_candidate = (
        ndwi.gt(ndwi_threshold)
        .And(mndwi.gt(mndwi_threshold))
    )

    water_candidate = (
        water_candidate
        .Or(
            awei.gt(awei_threshold)
            .And(ndwi.gt(ndwi_secondary_threshold))
        )
    )

    water_mask = (
        water_candidate
        .And(vegetation_free)
        .And(builtup_free)
    )


    # ============================================================
    # 10. CLEAN WATER MASK (MORPHOLOGY)
    #
    # A small closing (dilate then erode) fills tiny gaps/noise
    # inside water bodies while preserving shoreline shape better
    # than a large erosion/dilation pair would.
    # ============================================================

    cleaned = (
        water_mask
        .focal_max(
            radius=1,
            kernelType="circle",
            iterations=1
        )
        .focal_min(
            radius=1,
            kernelType="circle",
            iterations=1
        )
    )


    # ============================================================
    # 11. REMOVE SMALL CONNECTED PATCHES
    # ============================================================

    connected = (
        cleaned.connectedPixelCount(
            100,
            False
        )
    )

    cleaned = (
        cleaned.updateMask(
            connected.gte(min_connected_pixels)
        )
    )


    # ============================================================
    # 12. CONVERT WATER TO POLYGONS
    #
    # scale=10 sharpens the boundary using the 10 m-native bands;
    # see BOUNDARY_SCALE_M note above for why this is still valid
    # even though MNDWI/AWEI depend on 20 m SWIR bands.
    # ============================================================

    vectors_raw = (
        cleaned
        .selfMask()
        .reduceToVectors(
            geometry=roi,
            scale=BOUNDARY_SCALE_M,
            geometryType="polygon",
            labelProperty="water",
            maxPixels=int(1e9),
            bestEffort=True,
            tileScale=4
        )
    )


    # ============================================================
    # 13. BOUNDARY REFINEMENT
    #
    # Very slight simplification only — enough to remove raster
    # staircasing artifacts without turning natural shorelines
    # into artificially geometric shapes.
    # ============================================================

    def simplify_feature(feature):
        return feature.setGeometry(
            feature.geometry().simplify(
                SIMPLIFY_TOLERANCE_M
            )
        )

    vectors = vectors_raw.map(simplify_feature)


    # ============================================================
    # 14. CALCULATE AREA
    # ============================================================

    def add_area(feature):

        area = feature.geometry().area(
            1
        )

        return feature.set({

            "area_m2":
                area,

            "area_km2":
                area.divide(
                    1_000_000
                )

        })


    vectors = vectors.map(
        add_area
    )


    # ============================================================
    # 15. REMOVE SMALL WATER BODIES
    # ============================================================

    vectors = (
        vectors.filter(
            ee.Filter.gt(
                "area_m2",
                area_min
            )
        )
    )


    # ============================================================
    # 16. SATELLITE VISUALIZATION
    #
    # IMPORTANT:
    # Use display_roi, NOT roi.
    #
    # Therefore satellite imagery continues outside the
    # water-detection circle.
    # ============================================================

    satellite_display = (
        s2
        .clip(display_roi)
    )

    s2_map = (
        satellite_display
        .visualize(
            bands=[
                "B4",
                "B3",
                "B2"
            ],
            min=0,
            max=3000,
            gamma=1.2
        )
        .getMapId()
    )


    # ============================================================
    # 17. FALSE COLOR COMPOSITE (FCC)
    #
    # NIR-Red-Green. This is a visual analysis aid only — it does
    # not feed the classification decision, which is driven by the
    # spectral indices above.
    # ============================================================

    fcc_map = (
        satellite_display
        .visualize(
            bands=[
                "B8",
                "B4",
                "B3"
            ],
            min=0,
            max=3000,
            gamma=1.2
        )
        .getMapId()
    )


    # ============================================================
    # 18. NDWI VISUALIZATION
    #
    # NDWI remains limited to detection region.
    # ============================================================

    ndwi_map = (
        ndwi
        .visualize(
            min=-0.3,
            max=0.5,
            palette=[
                "#8B4513",
                "#DAA520",
                "#228B22",
                "#00CED1",
                "#0000FF"
            ]
        )
        .getMapId()
    )


    # ============================================================
    # 19. WATER MASK VISUALIZATION
    # ============================================================

    mask_map = (
        cleaned
        .selfMask()
        .visualize(
            palette=[
                "#00BFFF"
            ]
        )
        .getMapId()
    )


    # ============================================================
    # 20. TILE URL
    # ============================================================

    def tile_url(map_object):

        return (
            "https://earthengine.googleapis.com/v1/"
            + map_object["mapid"]
            + "/tiles/{z}/{x}/{y}"
        )


    # ============================================================
    # 21. GET GEOJSON
    #
    # This is the only large .getInfo() call in the pipeline —
    # everything upstream stays server-side to keep this fast even
    # at the 100 km radius option.
    # ============================================================

    raw = vectors.getInfo()


    if (
        isinstance(raw, dict)
        and raw.get("type")
        == "FeatureCollection"
    ):

        features = raw.get(
            "features",
            []
        )

    elif (
        isinstance(raw, dict)
        and "features" in raw
    ):

        features = raw[
            "features"
        ]

    elif isinstance(raw, list):

        features = raw

    else:

        features = []


    # ============================================================
    # 22. CLEAN + NUMBER WATER BODIES
    # ============================================================

    clean_features = []


    for feature in features:

        if not isinstance(
            feature,
            dict
        ):
            continue


        geometry = feature.get(
            "geometry"
        )


        if not geometry:
            continue


        if not geometry.get(
            "type"
        ):
            continue


        if not geometry.get(
            "coordinates"
        ):
            continue


        properties = feature.get(
            "properties",
            {}
        ).copy()


        # --------------------------------------------------------
        # WATER BODY NUMBER
        # --------------------------------------------------------

        water_body_id = (
            len(clean_features) + 1
        )


        properties[
            "water_body_id"
        ] = water_body_id


        properties[
            "water_body_name"
        ] = (
            f"Water Body "
            f"{water_body_id}"
        )


        # --------------------------------------------------------
        # AREA
        # --------------------------------------------------------

        try:

            area_m2 = float(
                properties.get(
                    "area_m2",
                    0
                )
            )

        except (
            TypeError,
            ValueError
        ):

            area_m2 = 0


        properties[
            "area_m2"
        ] = area_m2


        properties[
            "area_km2"
        ] = (
            area_m2
            / 1_000_000
        )


        clean_features.append({

            "type":
                "Feature",

            "geometry":
                geometry,

            "properties":
                properties

        })


    # ============================================================
    # 23. FINAL GEOJSON
    # ============================================================

    final_geojson = {

        "type":
            "FeatureCollection",

        "features":
            clean_features

    }


    # ============================================================
    # 24. RETURN
    # ============================================================

    return {

        "geojson":
            final_geojson,

        "feature_count":
            len(
                clean_features
            ),

        "center": {

            "latitude":
                latitude,

            "longitude":
                longitude

        },

        "radius_km":
            radius_km,

        # Detection/display bounds.
        # Frontend can use these for map fitting.
        "bbox":
            display_bounds,

        "parameters": {

            "ndwi_threshold":
                ndwi_threshold,

            "ndwi_secondary_threshold":
                ndwi_secondary_threshold,

            "mndwi_threshold":
                mndwi_threshold,

            "awei_threshold":
                awei_threshold,

            "max_ndvi":
                max_ndvi,

            "max_ndbi":
                max_ndbi,

            "min_connected_pixels":
                min_connected_pixels,

            "area_min":
                area_min,

        },

        "tile_urls": {

            "satellite":
                tile_url(
                    s2_map
                ),

            "fcc":
                tile_url(
                    fcc_map
                ),

            "ndwi":
                tile_url(
                    ndwi_map
                ),

            "water_mask":
                tile_url(
                    mask_map
                )

        }

    }
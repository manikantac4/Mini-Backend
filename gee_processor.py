import ee


# ================================================================
# WATER BODY PROCESSOR
# ================================================================

def process_water_boundaries(
    latitude,
    longitude,
    radius_km=50,
    threshold=0.1,
    area_min=8000
):

    # ============================================================
    # 1. CENTER POINT
    # ============================================================

    latitude = float(latitude)
    longitude = float(longitude)
    radius_km = float(radius_km)

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
    # ============================================================

    collection = (
        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
        )
        .filterBounds(roi)
        .filterDate(
            "2023-11-01",
            "2024-03-31"
        )
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                20
            )
        )
    )


    # ============================================================
    # 5. CLOUD / SHADOW MASK
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


    # ============================================================
    # 6. CREATE SENTINEL COMPOSITE
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
    # 8. WATER INDICES
    # ============================================================

    ndwi = (
        s2_detection
        .normalizedDifference([
            "B3",
            "B8"
        ])
        .rename("NDWI")
    )

    mndwi = (
        s2_detection
        .normalizedDifference([
            "B3",
            "B11"
        ])
        .rename("MNDWI")
    )


    # ============================================================
    # 9. WATER MASK
    # ============================================================

    water_mask = (
        ndwi
        .gt(float(threshold))
        .Or(
            mndwi.gt(0.0)
        )
        .And(
            ndwi.gt(-0.2)
        )
        .And(
            mndwi.gt(-0.2)
        )
    )


    # ============================================================
    # 10. CLEAN WATER MASK
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


    # ============================================================
    # 11. REMOVE SMALL CONNECTED PATCHES
    # ============================================================

    connected = (
        cleaned.connectedPixelCount(
            50,
            False
        )
    )

    cleaned = (
        cleaned.updateMask(
            connected.gte(15)
        )
    )


    # ============================================================
    # 12. CONVERT WATER TO POLYGONS
    # ============================================================

    vectors = (
        cleaned
        .selfMask()
        .reduceToVectors(
            geometry=roi,
            scale=20,
            geometryType="polygon",
            labelProperty="water",
            maxPixels=int(1e9),
            bestEffort=True,
            tileScale=4
        )
    )


    # ============================================================
    # 13. CALCULATE AREA
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
    # 14. REMOVE SMALL WATER BODIES
    # ============================================================

    vectors = (
        vectors.filter(
            ee.Filter.gt(
                "area_m2",
                float(area_min)
            )
        )
    )


    # ============================================================
    # 15. SATELLITE VISUALIZATION
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
    # 16. NDWI VISUALIZATION
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
    # 17. WATER MASK VISUALIZATION
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
    # 18. TILE URL
    # ============================================================

    def tile_url(map_object):

        return (
            "https://earthengine.googleapis.com/v1/"
            + map_object["mapid"]
            + "/tiles/{z}/{x}/{y}"
        )


    # ============================================================
    # 19. GET GEOJSON
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
    # 20. CLEAN + NUMBER WATER BODIES
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
    # 21. FINAL GEOJSON
    # ============================================================

    final_geojson = {

        "type":
            "FeatureCollection",

        "features":
            clean_features

    }


    # ============================================================
    # 22. RETURN
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

        "tile_urls": {

            "satellite":
                tile_url(
                    s2_map
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
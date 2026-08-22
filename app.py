from flask import Flask, request, jsonify
from flask_cors import CORS

import ee
import json
import os

from gee_processor import process_water_boundaries
from services.geocoder import get_location
from services.gee_service import get_sentinel_image


# ================================================================
# EARTH ENGINE INITIALIZATION
# ================================================================

service_account_json = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS_JSON"
)

if service_account_json:

    # ------------------------------------------------------------
    # RENDER / PRODUCTION
    # ------------------------------------------------------------

    service_account_info = json.loads(
        service_account_json
    )

    credentials = ee.ServiceAccountCredentials(
        service_account_info["client_email"],
        key_data=json.dumps(
            service_account_info
        )
    )

    ee.Initialize(
        credentials=credentials,
        project="water-segmentation-gee"
    )

else:

    # ------------------------------------------------------------
    # LOCAL DEVELOPMENT
    # Uses:
    # gcloud auth application-default login
    # ------------------------------------------------------------

    ee.Initialize(
        project="water-segmentation-gee"
    )


# ================================================================
# FLASK
# ================================================================

app = Flask(__name__)

CORS(app)


# ================================================================
# HOME
# ================================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "status": "ok",

        "message":
            "Water Detection API running"

    })


# ================================================================
# DETECT WATER
# ================================================================

@app.route(
    "/detect-water",
    methods=["POST"]
)
def detect_water():

    try:

        body = request.get_json() or {}

        # --------------------------------------------------------
        # INPUT
        # --------------------------------------------------------

        latitude = body.get(
            "latitude"
        )

        longitude = body.get(
            "longitude"
        )

        radius_km = body.get(
            "radius_km",
            50
        )

        threshold = body.get(
            "threshold",
            0.1
        )

        area_min = body.get(
            "area_min",
            8000
        )

        # Advanced / optional tuning parameters. These are all
        # optional — if the frontend doesn't send them, the
        # defaults defined in gee_processor.py are used.

        ndwi_secondary_threshold = body.get(
            "ndwi_secondary_threshold"
        )

        mndwi_threshold = body.get(
            "mndwi_threshold"
        )

        awei_threshold = body.get(
            "awei_threshold"
        )

        max_ndvi = body.get(
            "max_ndvi"
        )

        max_ndbi = body.get(
            "max_ndbi"
        )

        min_connected_pixels = body.get(
            "min_connected_pixels"
        )


        # --------------------------------------------------------
        # VALIDATE COORDINATES
        # --------------------------------------------------------

        if (
            latitude is None
            or longitude is None
        ):

            return jsonify({

                "status": "error",

                "message":
                    "latitude and longitude are required"

            }), 400


        # --------------------------------------------------------
        # CONVERT NUMERIC VALUES
        # --------------------------------------------------------

        try:

            latitude = float(
                latitude
            )

            longitude = float(
                longitude
            )

            radius_km = float(
                radius_km
            )

            threshold = float(
                threshold
            )

            area_min = float(
                area_min
            )

            # Optional advanced parameters — only cast if provided.

            if ndwi_secondary_threshold is not None:
                ndwi_secondary_threshold = float(ndwi_secondary_threshold)

            if mndwi_threshold is not None:
                mndwi_threshold = float(mndwi_threshold)

            if awei_threshold is not None:
                awei_threshold = float(awei_threshold)

            if max_ndvi is not None:
                max_ndvi = float(max_ndvi)

            if max_ndbi is not None:
                max_ndbi = float(max_ndbi)

            if min_connected_pixels is not None:
                min_connected_pixels = int(min_connected_pixels)

        except (TypeError, ValueError):

            return jsonify({

                "status": "error",

                "message":
                    "Invalid numeric input"

            }), 400


        # --------------------------------------------------------
        # COORDINATE RANGE
        # --------------------------------------------------------

        if not -90 <= latitude <= 90:

            return jsonify({

                "status": "error",

                "message":
                    "Invalid latitude"

            }), 400


        if not -180 <= longitude <= 180:

            return jsonify({

                "status": "error",

                "message":
                    "Invalid longitude"

            }), 400


        # --------------------------------------------------------
        # ALLOWED SCAN RADII
        # --------------------------------------------------------

        allowed_radii = [
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100
        ]


        # --------------------------------------------------------
        # RADIUS VALIDATION
        # --------------------------------------------------------

        if radius_km not in allowed_radii:

            return jsonify({

                "status": "error",

                "message":
                    "radius_km must be one of: "
                    "30, 40, 50, 60, 70, 80, 90, 100 km",

                "allowed_radii":
                    allowed_radii

            }), 400


        # --------------------------------------------------------
        # WATER DETECTION
        #
        # Only pass advanced parameters that were actually supplied
        # so gee_processor.py's own defaults apply otherwise.
        # --------------------------------------------------------

        detection_kwargs = {

            "latitude": latitude,

            "longitude": longitude,

            "radius_km": radius_km,

            "threshold": threshold,

            "area_min": area_min,

        }

        optional_params = {

            "ndwi_secondary_threshold": ndwi_secondary_threshold,

            "mndwi_threshold": mndwi_threshold,

            "awei_threshold": awei_threshold,

            "max_ndvi": max_ndvi,

            "max_ndbi": max_ndbi,

            "min_connected_pixels": min_connected_pixels,

        }

        for key, value in optional_params.items():

            if value is not None:
                detection_kwargs[key] = value

        result = process_water_boundaries(
            **detection_kwargs
        )


        # --------------------------------------------------------
        # RESPONSE
        # --------------------------------------------------------

        return jsonify({

            "status":
                "success",

            "feature_count":
                result["feature_count"],

            "center":
                result["center"],

            "radius_km":
                result["radius_km"],

            "bbox":
                result["bbox"],

            "geojson":
                result["geojson"],

            "tile_urls":
                result["tile_urls"],

            "parameters":
                result.get("parameters", {})

        })


    except Exception as e:

        print(
            "DETECT WATER ERROR:",
            str(e)
        )

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 500


# ================================================================
# ANALYZE PLACE
# ================================================================

@app.route(
    "/ai/analyze-place",
    methods=["POST"]
)
def analyze_place():

    try:

        body = request.get_json() or {}

        place_name = body.get(
            "place"
        )


        if not place_name:

            return jsonify({

                "status": "error",

                "message":
                    "place is required"

            }), 400


        location = get_location(
            place_name
        )


        if not location:

            return jsonify({

                "status": "error",

                "message":
                    "place not found"

            }), 404


        image = get_sentinel_image(

            location["lat"],

            location["lon"]

        )


        return jsonify({

            "status":
                "success",

            "phase":
                2,

            "place":
                place_name,

            "latitude":
                location["lat"],

            "longitude":
                location["lon"],

            "image_found":
                image is not None

        })


    except Exception as e:

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 500


# ================================================================
# LOCAL SERVER
# ================================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )
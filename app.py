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
        # --------------------------------------------------------

        result = process_water_boundaries(

            latitude=latitude,

            longitude=longitude,

            radius_km=radius_km,

            threshold=threshold,

            area_min=area_min

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
                result["tile_urls"]

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
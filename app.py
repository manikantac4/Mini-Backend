from flask import Flask, request, jsonify
from flask_cors import CORS
import ee
import json
import os

from gee_processor import process_water_boundaries
from services.geocoder import get_location
from services.gee_service import get_sentinel_image


# ============================================================
# GOOGLE EARTH ENGINE INITIALIZATION
# ============================================================
#
# LOCAL WINDOWS:
#   Uses Google Application Default Credentials (ADC)
#
# RENDER:
#   Uses GOOGLE_APPLICATION_CREDENTIALS_JSON
#
# ============================================================

if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON"):

    # --------------------------------------------------------
    # Render / Production
    # --------------------------------------------------------
    service_account_info = json.loads(
        os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
    )

    credentials = ee.ServiceAccountCredentials(
        service_account_info["client_email"],
        key_data=json.dumps(service_account_info)
    )

    ee.Initialize(
        credentials=credentials,
        project="water-segmentation-gee"
    )

else:

    # --------------------------------------------------------
    # Local Windows Development
    # --------------------------------------------------------
    ee.Initialize(
        project="water-segmentation-gee"
    )


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "ok",
        "message": "Water Detection API running"
    })


# ============================================================
# WATER DETECTION
# ============================================================

@app.route("/detect-water", methods=["POST"])
def detect_water():

    try:

        body = request.get_json()

        if not body:
            return jsonify({
                "status": "error",
                "message": "Request body is required"
            }), 400

        bbox = body.get("bbox")

        if not bbox or len(bbox) != 4:

            return jsonify({
                "status": "error",
                "message": "bbox required: [west, south, east, north]"
            }), 400

        west, south, east, north = bbox

        if west >= east or south >= north:

            return jsonify({
                "status": "error",
                "message": "Invalid bbox"
            }), 400

        threshold = body.get("threshold", 0.1)
        area_min = body.get("area_min", 8000)

        result = process_water_boundaries(
            bbox,
            threshold,
            area_min
        )

        return jsonify({

            "status": "success",

            "feature_count":
                result["feature_count"],

            "bbox":
                bbox,

            "geojson":
                result["geojson"],

            "tile_urls":
                result["tile_urls"]

        })

    except Exception as e:

        return jsonify({

            "status": "error",

            "message":
                str(e)

        }), 500


# ============================================================
# AI ANALYZE PLACE
# ============================================================

@app.route("/ai/analyze-place", methods=["POST"])
def analyze_place():

    try:

        body = request.get_json()

        if not body:

            return jsonify({
                "status": "error",
                "message": "Request body is required"
            }), 400

        place_name = body.get("place")

        if not place_name:

            return jsonify({
                "status": "error",
                "message": "place is required"
            }), 400

        location = get_location(place_name)

        if not location:

            return jsonify({
                "status": "error",
                "message": "place not found"
            }), 404

        image = get_sentinel_image(
            location["lat"],
            location["lon"]
        )

        return jsonify({

            "status": "success",

            "phase": 2,

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

            "message":
                str(e)

        }), 500


# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )
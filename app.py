from flask import Flask, request, jsonify
from flask_cors import CORS
import ee

from gee_processor import process_water_boundaries
from services.geocoder import get_location
from services.gee_service import get_sentinel_image


# ─────────────────────────────────────────────────────────────
# Initialize Google Earth Engine
# Uses Google Application Default Credentials (ADC)
# configured on this Windows machine.
# ─────────────────────────────────────────────────────────────
ee.Initialize(project="water-segmentation-gee")


app = Flask(__name__)
CORS(app)


# ─────────────────────────────────────────────────────────────
# Home
# ─────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "Water Detection API running"
    })


# ─────────────────────────────────────────────────────────────
# Water Detection
# ─────────────────────────────────────────────────────────────
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
                "error": "bbox required: [west, south, east, north]"
            }), 400

        west, south, east, north = bbox

        if west >= east or south >= north:
            return jsonify({
                "error": "Invalid bbox"
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
            "feature_count": result["feature_count"],
            "bbox": bbox,
            "geojson": result["geojson"],
            "tile_urls": result["tile_urls"]
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ─────────────────────────────────────────────────────────────
# AI Analyze Place
# ─────────────────────────────────────────────────────────────
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
            "place": place_name,
            "latitude": location["lat"],
            "longitude": location["lon"],
            "image_found": image is not None
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ─────────────────────────────────────────────────────────────
# Run Flask
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(
        debug=True,
        port=5000
    )
from flask import Flask, request, jsonify
from flask_cors import CORS
import ee
from gee_processor import process_water_boundaries

ee.Initialize(project='water-body-extraction-490703')

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "Water Detection API running"})

@app.route("/detect-water", methods=["POST"])
def detect_water():
    try:
        body      = request.get_json()
        bbox      = body.get("bbox")
        if not bbox or len(bbox) != 4:
            return jsonify({"error": "bbox required: [west, south, east, north]"}), 400

        west, south, east, north = bbox
        if west >= east or south >= north:
            return jsonify({"error": "Invalid bbox"}), 400

        threshold = body.get("threshold", 0.1)
        area_min  = body.get("area_min",  8000)

        result = process_water_boundaries(bbox, threshold, area_min)

        return jsonify({
            "status":        "success",
            "feature_count": result["feature_count"],
            "bbox":          bbox,
            "geojson":       result["geojson"],
            "tile_urls":     result["tile_urls"],   # ← new
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
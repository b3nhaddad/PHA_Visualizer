from flask import Blueprint, jsonify, request

p2_bp = Blueprint("p2", __name__)

@app.route("/state", methods=["GET"])
def get_state():
    return jsonify({"state": get_state()})

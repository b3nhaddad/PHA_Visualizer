from flask import Blueprint, jsonify, request
from math import ceil, sqrt
import torch
from api.schemas import paramaters_pure

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trajectories = []
_state = {}

p2_bp = Blueprint("p2", __name__)

@p2_bp.route("/state", methods=["GET"])
def get_state():
    return jsonify({"state": _state})

@p2_bp.route("/state", methods=["POST"])
def set_state():
    global _state
    data = request.get_json()
    _state = paramaters_pure(data)
    return jsonify({"status": 200, "state": _state})


def spatial_hessian_descent_p2(step_size, dimensionality, start_position, hessian_fn, gradient_fn=None, n_steps=None):
    if n_steps is None:
        n_steps = ceil(1 / step_size)

    position = start_position.clone().to(device).flatten()
    N = dimensionality
    trajectory = [position.clone()]

    # Compute Hessian ONCE (constant for p=2)
    hess = hessian_fn(position)

    for i in range(n_steps):

        norm_sq = torch.dot(position, position)

        # Project Hessian to tangent space at current position
        if norm_sq > 1e-12:
            M = torch.eye(N, device=device) - torch.outer(position, position) / norm_sq
            hess_proj = M @ hess @ M
        else:
            hess_proj = hess

        # Most negative eigenvector of projected Hessian is already tangent
        _, eigenvectors = torch.linalg.eigh(hess_proj)
        v = eigenvectors[:, 0].clone()

        # Flip sign if gradient points same direction as v
        if gradient_fn is not None and torch.dot(v, gradient_fn(position)) > 0:
            v = -v

        position = (position + sqrt(N) * step_size * v).flatten()

        target_q = (i + 1) / n_steps
        target_radius = sqrt(N * target_q)
        position = (position / torch.linalg.norm(position) * target_radius).flatten()

        trajectory.append(position.clone())
        #

    return trajectory


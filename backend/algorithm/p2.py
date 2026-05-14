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
 

def spatial_hessian_descent_p2(step_size, dimensionality, start_position, hessian_fn, n_steps=None):
    if n_steps is None:
        n_steps = ceil(1 / step_size)

    position = start_position.clone().to(device).flatten()
    N = dimensionality #dimensionality
    trajectory = [position.clone()]
    prev_v = None

    for i in range(n_steps):
        hess = hessian_fn(position)

        #projects Hessian to orthogonal subspace of current position
        #this is needed because we can only move orthogonally to the sphere
        norm_sq = torch.dot(position, position)

        if norm_sq > 1e-12:
            M = torch.eye(N, device=device) - torch.outer(position, position) / norm_sq
            hess_proj = M @ hess @ M
            #projects to tangent space ensures that the only directions to move or orhtogonal
        else:
            hess_proj = hess

        # get eigenvalues and eigenvectors (sorted ascending, so index 0 is most negative)
        eigenvalues, eigenvectors = torch.linalg.eigh(hess_proj)

        # Find most negative eigenvector orthogonal to position
        v = eigenvectors[:, 0].clone()

        # Enforce sign consistency with previous step to prevent oscillation.
        # gradient_fn check doesn't work for p=2 because H@sigma = lambda*sigma
        # (parallel to sigma, zero dot product with all tangent vectors).
        if prev_v is not None and torch.dot(v, prev_v) < 0:
            v = -v
        prev_v = v.clone()

        position = (position + sqrt(N) * step_size * v).flatten()

        target_q = (i + 1) / n_steps
        target_radius = sqrt(N * target_q)
        position = (position / torch.linalg.norm(position) * target_radius).flatten()
        #goes back from tangent space to the sphere
        trajectory.append(position.clone())
        #has to return current position

    return trajectory


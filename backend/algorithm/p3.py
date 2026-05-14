import time
from flask import Blueprint, jsonify, request
from math import ceil, sqrt
import torch
from api.schemas import paramaters_pure

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


p3_bp = Blueprint("p3", __name__)

def spatial_hessian_descentp3(step_size, dimensionality, start_position, hessian_fn, gradient_fn=None, n_steps=None, J=None):
    """
    This code uses the most negative eigenvector of the Hessian, since this value is 2v^T ∇^2Hₙ(σq)v is as negative as possible
    We do not use the Gradient of the hessian (first order derivative) because "we should have ∇HN (σq) ≈ 0, so we probably should not choose v
    based on the gradient." this is asserting using the first derivate of the Hessian will lead us to local minima and not the global minimum.

    Args:
        step_size: float, the step size (τ in the paper, typically 1/k),
        dimensionality: int, N dimension of the space
        start_position: array, starting point σ_0
        hessian_fn: function that returns the Hessian matrix at a point
        gradient_fn: optional, gradient function (used only to pick sign of direction)
        n_steps: int, number of steps k (default: ceil(1/step_size))

    Returns:
        trajectory: list of positions along the path
    """
    if n_steps is None:
        n_steps = ceil(1 / step_size)

    position = start_position.clone().to(device).flatten()
    N = dimensionality #dimensionality
    trajectory = [position.clone()]


    for i in range(n_steps):
        # Get current Hessian

        start = time.monotonic_ns()
        hess = hessian_fn(position)
        end = time.monotonic_ns()
        print(start - end)

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
        #hess_proj = torch.from_numpy(hess_proj)
        eigenvalues, eigenvectors = torch.linalg.eigh(hess_proj)


        #can also be replaced by PyHessian when in NN applications
        #top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues() <---
        #where       hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=True)

        # Find most negative eigenvector orthogonal to position
        v = eigenvectors[:, 0].clone()

        if norm_sq > 1e-12 and torch.dot(v, torch.einsum('ijk,j,k', J, position, position)) < 0:
          v = -v

        position = (position + sqrt(N) * step_size * v).flatten()

        target_q = (i + 1) / n_steps
        target_radius = sqrt(N * target_q)
        position = (position / torch.linalg.norm(position) * target_radius).flatten()
        #goes back from tangent space to the sphere
        trajectory.append(position.clone())
        #has to return current position

    return trajectory
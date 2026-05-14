import torch
import jax
import jax.numpy as jnp

def H_p2(s, J):
    #this is the summation of sigma i,j and J_i,_j
    #this is equivallent to -sigma transposer J sigma
    #J is the GOE and is centered at (0, 1/N) variance
    return -s @ J @ s


def hessian_fn(sigma,H):
    sigma = sigma.detach().flatten().requires_grad_(True)
    return torch.autograd.functional.hessian(H, sigma)

def H_p3(s,J):
    return -torch.einsum('ijk,i,j,k', J, s, s, s)


import torch
import jax
import jax.numpy as jnp

def H_p2(s, J):
    #this is the summation of sigma i,j and J_i,_j
    #this is equivallent to -sigma transposer J sigma
    #J is the GOE and is centered at (0, 1/N) variance
    return -s @ J @ s

def hessian_fn(H):
    #calculates the hessian of the hamiltonian at sigma
    hessian = jax.hessian(H)
    return hessian
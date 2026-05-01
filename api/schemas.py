#add schemas for visualizer

def paramaters_pure(state):
    step_size = state.get("step_size")
    dimensionality = state.get("step_size")
    start_position = state.get("step_size")
    hessian_fn = state.get("hessian_fn")
    gradient_fn = state.get("gradient_fn")
    n_steps = state.get("n_steps")
    return {"step_size": step_size, "dimensionality": dimensionality,
            "start_position": start_position, "hessian_fn": hessian_fn,
            "gradient_fn": gradient_fn, "n_steps": n_steps}
    #step_size, dimensionality, start_position, hessian_fn, gradient_fn = None, n_steps = None
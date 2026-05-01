#add schemas for visualizer

def paramaters_pure(state):
    step_size = state.get("step_size")
    dimensionality = state.get("dimensionality")
    start_position = state.get("start_position")
    n_steps = state.get("n_steps")
    return {"step_size": step_size, "dimensionality": dimensionality,
            "start_position": start_position, "n_steps": n_steps}
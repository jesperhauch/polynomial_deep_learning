from .modsim import State, System, TimeFrame

def make_system(beta: float, gamma: float, s:int, i:int, r:int, t_end:int) -> System:
    init = State(s=s, i=i, r=r)
    init /= init.sum()

    return System(init=init, t_end=t_end,
                  beta=beta, gamma=gamma)

def update_func(state: State, system: System) -> State: 
    s, i, r = state.s, state.i, state.r

    infected = system.beta * i * s    
    recovered = system.gamma * i
    
    s -= infected
    i += infected - recovered
    r += recovered
    
    return State(s=s, i=i, r=r)

def run_simulation(system, update_func):
    frame = TimeFrame(columns=system.init.index)
    frame.loc[0] = system.init
    
    for t in range(0, system.t_end):
        frame.loc[t+1] = update_func(frame.loc[t], system)
    
    return frame
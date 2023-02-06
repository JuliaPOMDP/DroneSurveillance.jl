function POMDPs.stateindex(mdp::DroneSurveillanceMDP, s::DSState)
    if isterminal(mdp, s)
        return length(mdp)
    end
    nx, ny = mdp.size 
    LinearIndices((nx, ny, nx, ny))[s.quad[1], s.quad[2], s.agent[1], s.agent[2]]
end

function state_from_index(mdp::DroneSurveillanceMDP, si::Int64)
    if si == length(mdp)
        return mdp.terminal_state
    end
    nx, ny = mdp.size 
    s = CartesianIndices((nx, ny, nx, ny))[si]
    return DSState([s[1], s[2]], [s[3], s[4]])
end

# the state space is the MDP itself
# we define an iterator over it

POMDPs.states(mdp::DroneSurveillanceMDP) = mdp
Base.length(mdp::DroneSurveillanceMDP) = (mdp.size[1] * mdp.size[2])^2 + 1

function Base.iterate(mdp::DroneSurveillanceMDP, i::Int64 = 1)
    if i > length(mdp)
        return nothing
    end
    s = state_from_index(mdp, i)
    return (s, i+1)
end

function POMDPs.initialstate(mdp::DroneSurveillanceMDP)
    quad = mdp.region_A
    nx, ny = mdp.size
    fov_x, fov_y = mdp.fov
    states = DSState[]
    if mdp.agent_policy == :restricted 
        xspace = fov_x:nx
        yspace = fov_y:ny
        for x in fov_x:nx
            for y in 1:ny
                agent = DSPos(x, y)
                push!(states, DSState(quad, agent))
            end
        end
        for y in fov_y:ny
            for x in 1:fov_x-1
                agent = DSPos(x, y)
                push!(states, DSState(quad, agent))
            end
        end

    else 
        for x in 1:nx 
            for y in 1:ny
                if (x,y) != (1,1)
                    agent = DSPos(x, y)
                    push!(states, DSState(quad, agent))
                end
            end
        end
    end
    probs = normalize!(ones(length(states)), 1)
    return SparseCat(states, probs)
end

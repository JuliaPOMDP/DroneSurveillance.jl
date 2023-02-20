function POMDPs.transition(mdp::DroneSurveillanceMDP, s::DSState, a::Int64) :: Union{Deterministic, SparseCat}
    if isterminal(mdp, s) || s.quad == s.agent
        return Deterministic(mdp.terminal_state) # the function is not type stable, returns either Deterministic or SparseCat
    end

    # move quad
    # if it would move out of bounds, just stay in place
    actor_inbounds(actor_state) = (0 < actor_state[1] <= mdp.size[1]) && (0 < actor_state[2] <= mdp.size[2])
    new_quad = actor_inbounds(s.quad + ACTION_DIRS[a]) ? s.quad + ACTION_DIRS[a] : s.quad

    # move agent 
    new_states = MVector{N_ACTIONS, DSState}(undef)
    probs = @MVector(zeros(N_ACTIONS))
    for (i, act) in enumerate(ACTION_DIRS)
        new_agent = actor_inbounds(s.agent + act) ? s.agent + act : s.agent
        if agent_inbounds(mdp, new_agent)
            new_states[i] = DSState(new_quad, new_agent)
            probs[i] += 1.0
        else
            new_states[i] = DSState(new_quad, s.agent)
        end
    end
    normalize!(probs, 1)
    return SparseCat(new_states, probs)
end

"""
    agent_inbounds(mdp::DroneSurveillanceMDP, s::DSPos)
returns true if s in an authorized position for the ground agent
s must be on the grid and outside of the surveyed regions
"""
function agent_inbounds(mdp::DroneSurveillanceMDP, s::DSPos)
    if !(0 < s[1] <= mdp.size[1]) || !(0 < s[2] <= mdp.size[2])
        return false
    end
    if mdp.agent_policy == :restricted 
        if s == mdp.region_A || s == mdp.region_B
            return false 
        end
    end
    return true
end
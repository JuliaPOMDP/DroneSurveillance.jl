function POMDPs.transition(pomdp::DroneSurveillancePOMDP, s::DSState, a::Int64)
    # move quad
    new_quad = s.quad + ACTION_DIRS[a]
    if !(0 < new_quad[1] <= pomdp.size[1]) || !(0 < new_quad[2] <= pomdp.size[2]) || isterminal(pomdp, s)
        return Deterministic(pomdp.terminal_state) # the function is not type stable, returns either Deterministic or SparseCat
    end

    # move agent 
    new_states = MVector{N_ACTIONS, DSState}(undef)
    probs = @MVector(zeros(N_ACTIONS))
    for (i, act) in enumerate(ACTION_DIRS)
        new_agent = s.agent + act
        new_states[i] = DSState(new_quad, new_agent)
        if agent_inbounds(pomdp, new_agent)
            probs[i] += 1.0
        end
    end
    normalize!(probs, 1)
    return SparseCat(new_states, probs)
end

"""
    agent_inbounds(pomdp::DroneSurveillancePOMDP, s::DSPos)
returns true if s in an authorized position for the ground agent
s must be on the grid and outside of the surveyed regions
"""
function agent_inbounds(pomdp::DroneSurveillancePOMDP, s::DSPos)
    if !(0 < s[1] <= pomdp.size[1]) || !(0 < s[2] <= pomdp.size[2])
        return false
    end
    if s == pomdp.region_A || s == pomdp.region_B
        return false 
    end
    return true
end
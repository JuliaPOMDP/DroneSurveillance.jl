function POMDPs.transition(pomdp::DroneSurveillancePOMDP, s::DSState, a::Int64)
    # move quad
    new_quad = s.quad + ACTION_DIRS[a]
    if !(0 < new_quad[1] <= pomdp.size[1]) || !(0 < new_quad[2] <= pomdp.size[2]) || isterminal(pomdp, s) || s.quad == s.agent
        return Deterministic(pomdp.terminal_state) # the function is not type stable, returns either Deterministic or SparseCat
    end

    # move agent 
    new_states = MVector{n_actions(pomdp), DSState}(undef)
    probs = @MVector(zeros(n_actions(pomdp)))
    for (i, act) in enumerate(ACTION_DIRS)
        new_agent = s.agent + act
        if agent_inbounds(pomdp, new_agent)
            new_states[i] = DSState(new_quad, new_agent)
            probs[i] += 1.0
        else
            new_states[i] = pomdp.terminal_state
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
    if pomdp.agent_policy == :restricted 
        if s == pomdp.region_A || s == pomdp.region_B
            return false 
        end
    end
    return true
end
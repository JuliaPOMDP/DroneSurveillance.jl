struct DSAgentStrat
    p :: Real
end
should_do_perfect_agent_step(agent::DSAgentStrat) = rand() <= agent.p

abstract type DSTransitionModel end
struct DSPerfectModel <: DSTransitionModel end
struct DSApproximateModel <: DSTransitionModel end


function POMDPs.transition(mdp::DroneSurveillanceMDP, s::DSState, a::DSPos) :: Union{Deterministic, SparseCat}
    return transition(mdp, mdp.agent_strategy, mdp.transition_model, s, a)
end

# for perfect model
function transition(mdp::DroneSurveillanceMDP, agent_strategy::DSAgentStrat, transition_model::DSPerfectModel, s::DSState, a::DSPos) :: Union{Deterministic, SparseCat}
    if isterminal(mdp, s) || s.quad == s.agent
        return Deterministic(mdp.terminal_state) # the function is not type stable, returns either Deterministic or SparseCat
    end

    # move quad
    # if it would move out of bounds, just stay in place
    actor_inbounds(actor_state) = (0 < actor_state[1] <= mdp.size[1]) && (0 < actor_state[2] <= mdp.size[2])
    new_quad = actor_inbounds(s.quad + a) ? s.quad + a : s.quad

    # move agent 
    new_states, probs = move_agent(mdp, agent_strategy, new_quad, s)
    return SparseCat(new_states, probs)
end

# for our approximate model
function transition(mdp::DroneSurveillanceMDP, agent_strategy::DSAgentStrat, transition_model::DSApproximateModel, s::DSState, a::DSPos) :: Union{Deterministic, SparseCat}
    if isterminal(mdp, s) || s.quad == s.agent
        return Deterministic(mdp.terminal_state) # the function is not type stable, returns either Deterministic or SparseCat
    end
    # TODO actually implement predict function
    new_states, probs = predict(transition_model, s, a)
    return SparseCat(new_states, probs)
end


function agent_optimal_action_idx(s::DSState) :: Int
    vec = s.quad - s.agent
    # similar to dot = atan2(vec'*[1; 0])
    angle = atan(vec[2] - 0, vec[1] - 1)
    if π/4 <= angle < π*3/4
        a = ACTIONS_DICT[:north]
    elseif -π/4 <= angle <= π/4
        a = ACTIONS_DICT[:east]
    elseif -π*3/4 <= angle < -π/4
        a = ACTIONS_DICT[:south]
    else 
        a = ACTIONS_DICT[:west]
    end
    return a
end

function move_agent(mdp::DroneSurveillanceMDP, agent_strategy::DSAgentStrat, new_quad, s::DSState)
    entity_inbounds(entity_state) = (0 < entity_state[1] <= mdp.size[1]) && (0 < entity_state[2] <= mdp.size[2])

    new_states = MVector{N_ACTIONS, DSState}(undef)
    probs = @MVector(zeros(N_ACTIONS))
    do_perfect_action = should_do_perfect_agent_step(agent_strategy)
    for (i, act) in enumerate(ACTION_DIRS)
        new_agent = entity_inbounds(s.agent + act) ? s.agent + act : s.agent
        if entity_inbounds(new_agent)
            new_states[i] = DSState(new_quad, new_agent)
            # Add extra probability to action in direction of drone
            if do_perfect_action 
                if act == ACTION_DIRS[agent_optimal_action_idx(s)]
                    probs[i] += 1.0
                end
            else
                probs[i] += 1.0
            end
        else
            new_states[i] = DSState(new_quad, s.agent)
            @assert false "We should never get here"
        end
    end
    normalize!(probs, 1)
    return new_states, probs
end

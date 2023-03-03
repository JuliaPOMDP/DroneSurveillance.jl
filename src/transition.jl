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
    if isterminal(mdp, s) || s.quad == s.agent || s.quad == mdp.region_B
        return Deterministic(mdp.terminal_state) # the function is not type stable, returns either Deterministic or SparseCat
    else
        # first, move quad
        # if it would move out of bounds, just stay in place
        actor_inbounds(actor_state) = (0 < actor_state[1] <= mdp.size[1]) && (0 < actor_state[2] <= mdp.size[2])
        new_quad = actor_inbounds(s.quad + a) ? s.quad + a : s.quad

        # then, move agent (independently)
        new_agent_distr = move_agent(mdp, agent_strategy, new_quad, s)
        if new_agent_distr isa SparseCat
            return SparseCat([DSState(new_quad, new_agent)
                              for new_agent in new_agent_distr.vals], new_agent_distr.probs)
        else
            return Deterministic(DSState(new_quad, new_agent_distr.val))
        end
    end
end

# for our approximate model
function transition(mdp::DroneSurveillanceMDP, agent_strategy::DSAgentStrat, transition_model::DSApproximateModel, s::DSState, a::DSPos) :: Union{Deterministic, SparseCat}
    if isterminal(mdp, s) || s.quad == s.agent || s.quad == mdp.region_B
        return Deterministic(mdp.terminal_state) # the function is not type stable, returns either Deterministic or SparseCat
    else
        new_states, probs = predict(transition_model, s, a)
        return SparseCat(new_states, probs)
    end
end


function agent_optimal_action_idx(s::DSState) :: Int
    vec = s.quad - s.agent
    # similar to dot = atan2(vec'*[1; 0])
    angle = atan(vec[2], vec[1])
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

function move_agent(mdp::DroneSurveillanceMDP, agent_strategy::DSAgentStrat, new_quad::DSPos, s::DSState)
    entity_inbounds(entity_state) = (0 < entity_state[1] <= mdp.size[1]) && (0 < entity_state[2] <= mdp.size[2])
    @assert entity_inbounds(s.agent) "Tried to move agent that's already out of bounds! $(s.agent), $(mdp.size)"

    if should_do_perfect_agent_step(agent_strategy)
        act_idx = agent_optimal_action_idx(s)
        act = ACTION_DIRS[act_idx]
        new_agent = entity_inbounds(s.agent + act) ? s.agent + act : s.agent
        return Deterministic(new_agent)
    else
        new_agent_states = MVector{N_ACTIONS, DSPos}(undef)
        probs = @MVector(zeros(N_ACTIONS))
        for (i, act) in enumerate(ACTION_DIRS)
            new_agent = entity_inbounds(s.agent + act) ? s.agent + act : s.agent
            if entity_inbounds(new_agent)
                new_agent_states[i] = new_agent
                # Add extra probability to action in direction of drone
                # just go randomly
                probs[i] += 1.0
            else
                @assert false "We should never get here. Maybe the agent was initialized out of bounds in the first place?"
            end
        end
        normalize!(probs, 1)
    end
    return SparseCat(new_agent_states, probs)
end

import Base: product
import Base

struct DSAgentStrat
    p :: Real
end
should_do_perfect_agent_step(agent::DSAgentStrat) = rand() <= agent.p

Base.:*(λ::Real, d::Deterministic) = SparseCat([d.val], [λ])
Base.:*(λ::Real, sc::SparseCat) = SparseCat(sc.vals, λ.*sc.probs)
"Add SparseCats (usually have to be multiplied by weight first)."
function ⊕(sc_lhs::SparseCat, sc_rhs::SparseCat)
    SparseCat(vcat(sc_lhs.vals, sc_rhs.vals),
              vcat(sc_lhs.probs, sc_rhs.probs))
end
"Multiply SparseCat"
⊗(d_lhs::Deterministic, sc_rhs::SparseCat) = SparseCat([d_lhs.val], [1]) ⊗ sc_rhs
⊗(sc_lhs::SparseCat, d_rhs::Deterministic) =  sc_lhs ⊗ SparseCat([d_rhs.val], [1])
function ⊗(sc_lhs::SparseCat, sc_rhs::SparseCat)
    vals = product(sc_lhs.vals, sc_rhs.vals) |> collect
    probs = map(prod, product(sc_lhs.probs, sc_rhs.probs)) |> collect
    return SparseCat(vals[:], probs[:])
end

abstract type DSTransitionModel end
struct DSPerfectModel <: DSTransitionModel end
struct DSApproximateModel <: DSTransitionModel end
struct DSLinModel{T} <: DSTransitionModel where T <: Real
    θ_Δx :: Matrix{T}
    θ_Δy :: Matrix{T}
end

function predict(model::DSLinModel, s::DSState, a::DSPos)
    nx, ny = length.([model.θ_Δx, model.θ_Δy]) .÷ 2
    states = (-nx:nx, -ny:ny) .|> collect

    Δx = s.agent.x - s.quad.x
    Δy = s.agent.y - s.quad.y
    ξ = [Δx, Δy, a.x, a.y, 1]
    softmax(x) = exp.(x) / sum(exp.(x))
    probs = (softmax(model.θ_Δx * ξ), softmax(model.θ_Δy * ξ))
    return SparseCat(states[1], probs[1]), SparseCat(states[2], probs[2])
end


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
        new_quad_distr = SparseCat([new_quad, s.quad], [3//4, 1//4])

        # then, move agent (independently)
        new_agent_distr = move_agent(mdp, agent_strategy, new_quad, s)

        new_state_dist = let new_state_distr = new_quad_distr ⊗ new_agent_distr
            states = [DSState(q, a) for (q, a) in new_state_distr.vals]
            SparseCat(states, new_state_distr.probs)
        end

        # TODO: probably we want to cull states with a probability < ϵ
        # and then re-normalize
        return new_state_dist
    end
end

# for our approximate model
function transition(mdp::DroneSurveillanceMDP, agent_strategy::DSAgentStrat, transition_model::DSLinModel, s::DSState, a::DSPos) :: Union{Deterministic, SparseCat}
    if isterminal(mdp, s) || s.quad == s.agent || s.quad == mdp.region_B
        return Deterministic(mdp.terminal_state) # the function is not type stable, returns either Deterministic or SparseCat
    else
        Δx_dist, Δy_dist = predict(transition_model, s, a)
        new_state_dist = let Δ_dist =  Δx_dist ⊗ Δy_dist
            # TODO: add possibility of no movement
            # with movement
            new_states_with_movement = 
                        [DSState(s.quad + a, s.quad + a + DSPos(Δ_quad_agent...))
                         for Δ_quad_agent in Δ_dist.vals]
            SparseCat(new_states, Δ_dist.probs)
            # without movement
            new_states_no_movement = 
                        [DSState(s.quad, s.quad + DSPos(Δ_quad_agent...))
                         for Δ_quad_agent in Δ_dist.vals]
            SparseCat(new_states_, Δ_dist.probs)
            (3//4 * new_states_with_movement ⊕ 1//4 * new_states_no_movement)
        end
        return new_state_dist
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

    perfect_agent = begin
        act_idx = agent_optimal_action_idx(s)
        act = ACTION_DIRS[act_idx]
        new_agent = entity_inbounds(s.agent + act) ? s.agent + act : s.agent
        Deterministic(new_agent)
    end
    random_agent = begin
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
        SparseCat(new_agent_states, probs)
    end
    return (agent_strategy.p*perfect_agent) ⊕ ((1-agent_strategy.p)*random_agent)
end

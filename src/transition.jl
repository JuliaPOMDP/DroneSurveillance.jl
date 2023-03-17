import Base: product
import Base
using LinearAlgebra: normalize, normalize!

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
    θ_Δx :: AbstractMatrix{T}
    θ_Δy :: AbstractMatrix{T}
end
mutable struct DSLinCalModel{T} <: DSTransitionModel where T <: Real
    lin_model :: DSLinModel{T}
    temperature :: Float64
end
struct DSConformalizedModel{T} <: DSTransitionModel where T <: Real
    lin_model :: DSLinModel{T}
    conf_map_Δx :: Dict{Float64, Float64}
    conf_map_Δy :: Dict{Float64, Float64}
end

function prune_states(sc::SparseCat, ϵ_prune)
    idx = sc.probs .>= ϵ_prune
    SparseCat(sc.vals[idx], normalize(sc.probs[idx], 1))
end

# TODO maybe move this to the other project
function predict(model::DSLinModel, s::DSState, a::DSPos; ϵ_prune=1e-4, T=1.0)
    nx, ny = size.([model.θ_Δx, model.θ_Δy], 1) .÷ 2
    states_Δx, states_Δy = (-nx:nx, -ny:ny) .|> collect

    Δx = s.agent.x - s.quad.x
    Δy = s.agent.y - s.quad.y
    ξ = [Δx, Δy, a.x, a.y, 1]
    softmax(x) = exp.(x./T) / sum(exp.(x./T))
    probs_Δx, probs_Δy = (softmax(model.θ_Δx * ξ),
                          softmax(model.θ_Δy * ξ))

    # we prune states with small probability
    return (prune_states(SparseCat(states_Δx, probs_Δx), ϵ_prune),
            prune_states(SparseCat(states_Δy, probs_Δy), ϵ_prune))
end

function predict(cal_model::DSLinCalModel, s::DSState, a::DSPos; ϵ_prune=1e-4)
    lin_model = cal_model.lin_model
    T = cal_model.temperature

    nx, ny = size.([lin_model.θ_Δx, lin_model.θ_Δy], 1) .÷ 2
    states_Δx, states_Δy = (-nx:nx, -ny:ny) .|> collect

    Δx = s.agent.x - s.quad.x
    Δy = s.agent.y - s.quad.y
    ξ = [Δx, Δy, a.x, a.y, 1]
    softmax(x) = exp.(x / T) / sum(exp.(x / T))
    probs_Δx, probs_Δy = (softmax(lin_model.θ_Δx * ξ),
                          softmax(lin_model.θ_Δy * ξ))

    # we prune states with small probability
    return (prune_states(SparseCat(states_Δx, probs_Δx), ϵ_prune),
            prune_states(SparseCat(states_Δy, probs_Δy), ϵ_prune))
end

# make a prediction set with the linear model
function predict(model::DSLinModel, s::DSState, a::DSPos, λ::Real; ϵ_prune=1e-4)
    lhs_distr, rhs_distr = predict(model, s, a; ϵ_prune=ϵ_prune)

    # Shuffle predictions, keep adding to prediction set until just over or just under
    # desired probability (whichever has smaller "gap" to λ).
    lhs_pred_set, rhs_pred_set = Tuple([begin
            perm = shuffle(eachindex(distr.probs))
            p_perm = distr.probs[perm]
            p_cum = cumsum(p_perm)

            idx = begin
                idx = findfirst(>=(λ), p_cum)
                gap_hi = p_cum[idx] - λ
                gap_lo = λ - get(p_cum, idx-1, 0)
                (gap_hi < gap_lo ? idx : idx-1)
            end

            val_perm = distr.vals[perm]
            Set(val_perm[1:idx])
        end
        for distr in [lhs_distr, rhs_distr]
    ])
    return lhs_pred_set, rhs_pred_set
end

function predict(cal_model::DSLinCalModel, s::DSState, a::DSPos, λ::Real; ϵ_prune=1e-4)
    lhs_distr, rhs_distr = predict(cal_model, s, a; ϵ_prune=ϵ_prune)

    # Shuffle predictions, keep adding to prediction set until just over or just under
    # desired probability (whichever has smaller "gap" to λ).
    lhs_pred_set, rhs_pred_set = Tuple([begin
            perm = shuffle(eachindex(distr.probs))
            p_perm = distr.probs[perm]
            p_cum = cumsum(p_perm)

            idx = begin
                idx = findfirst(>=(λ), p_cum)
                gap_hi = p_cum[idx] - λ
                gap_lo = λ - get(p_cum, idx-1, 0)
                (gap_hi < gap_lo ? idx : idx-1)
            end

            val_perm = distr.vals[perm]
            Set(val_perm[1:idx])
        end
        for distr in [lhs_distr, rhs_distr]
    ])
    return lhs_pred_set, rhs_pred_set
end

function predict(conf_model::DSConformalizedModel, s::DSState, a::DSPos, λ::Real; ϵ_prune=1e-4)
    lin_model = conf_model.lin_model
    nx, ny = size.([lin_model.θ_Δx, lin_model.θ_Δy], 1) .÷ 2
    states = (-nx:nx, -ny:ny) .|> collect

    Δx = s.agent.x - s.quad.x
    Δy = s.agent.y - s.quad.y
    ξ = [Δx, Δy, a.x, a.y, 1]
    softmax(x) = exp.(x) / sum(exp.(x))
    probs = (softmax(lin_model.θ_Δx * ξ), softmax(lin_model.θ_Δy * ξ))
    λ_hat_Δx = conf_model.conf_map_Δx[λ]
    λ_hat_Δy = conf_model.conf_map_Δy[λ]

    idx_Δx = probs[1] .>= (1-λ_hat_Δx)
    idx_Δy = probs[2] .>= (1-λ_hat_Δy)
    pred_set_Δx = states[1][idx_Δx] |> Set
    pred_set_Δy = states[2][idx_Δy] |> Set
    return (pred_set_Δx, pred_set_Δy)
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

# for our linear and linear calibrated model
function transition(mdp::DroneSurveillanceMDP, agent_strategy::DSAgentStrat, transition_model::Union{DSLinModel, DSLinCalModel}, s::DSState, a::DSPos) :: Union{Deterministic, SparseCat}
    if isterminal(mdp, s) || s.quad == s.agent || s.quad == mdp.region_B
        return Deterministic(mdp.terminal_state) # the function is not type stable, returns either Deterministic or SparseCat
    else
        Δx_dist, Δy_dist = predict(transition_model, s, a)
        new_state_dist = let Δ_dist = Δx_dist ⊗ Δy_dist
            # the agent stays in place with chance 1/4
            new_states_with_movement = begin
                new_states = [DSState(s.quad + a, s.quad + a + DSPos(Δ_quad_agent...))
                              for Δ_quad_agent in Δ_dist.vals]
                SparseCat(new_states, Δ_dist.probs)
            end
            new_states_no_movement = begin
                new_states = [DSState(s.quad, s.quad + DSPos(Δ_quad_agent...))
                              for Δ_quad_agent in Δ_dist.vals]
                SparseCat(new_states, Δ_dist.probs)
            end
            (3//4 * new_states_with_movement ⊕ 1//4 * new_states_no_movement)
        end
        return new_state_dist
    end
end

# for our conformalized model
function transition(mdp::DroneSurveillanceMDP, agent_strategy::DSAgentStrat, transition_model::DSConformalizedModel, s::DSState, a::DSPos, λ::Real)
    if isterminal(mdp, s) || s.quad == s.agent || s.quad == mdp.region_B
        return Deterministic(mdp.terminal_state) # the function is not type stable, returns either Deterministic or SparseCat
    else
        return predict(transition_model, s, a, λ)
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
        @assert entity_inbounds(new_agent) "Somehow the new agent is out of bounds??"
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

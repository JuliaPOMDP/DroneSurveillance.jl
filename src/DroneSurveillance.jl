module DroneSurveillance

using Random
using LinearAlgebra
using POMDPs
using POMDPModelTools
using Parameters
using StaticArrays
using Compose
using Colors

export 
    DSPos,
    DSState,
    DroneSurveillancePOMDP


const DSPos = SVector{2, Int64}

struct DSState
    quad::DSPos
    agent::DSPos
end

"""
    DroneSurveillancePOMDP <: POMDP{DSState, Int64, Int64}

# Fields 
- `size::Tuple{Int64, Int64} = (5,5)` size of the grid world
- `region_A::DSPos = [1, 1]` first region to survey, initial state of the quad
- `region_B::DSPos = [size[1], size[2]]` second region to survey
- `fov::Tuple{Int64, Int64} = (3, 3)` size of the field of view of the drone
- `agent_policy::Symbol = :random` policy of the other agent, only random is implemented
- `terminal_state::DSState = DSState([-1, -1], [-1, -1])` a sentinel state to encode terminal states
- `discount_factor::Float64 = 0.95` the discount factor
"""
@with_kw mutable struct DroneSurveillancePOMDP <: POMDP{DSState, Int64, Int64}
    size::Tuple{Int64, Int64} = (5,5)
    region_A::DSPos = [1, 1]
    region_B::DSPos = [size[1], size[2]]
    fov::Tuple{Int64, Int64} = (3, 3)
    agent_policy::Symbol = :random
    terminal_state::DSState = DSState([-1, -1], [-1, -1])
    discount_factor::Float64 = 0.95
end

POMDPs.isterminal(pomdp::DroneSurveillancePOMDP, s::DSState) = s == pomdp.terminal_state 
POMDPs.discount(pomdp::DroneSurveillancePOMDP) = pomdp.discount_factor

function POMDPs.reward(pomdp::DroneSurveillancePOMDP, s::DSState, a::Int64, sp::DSState)
    if sp.quad == sp.agent
        return -1.0
    end
    if sp.quad == pomdp.region_B
        return 1.0
    end
    return 0.0
end

POMDPs.reward(pomdp::DroneSurveillancePOMDP, s::DSState, a::Int64) = reward(pomdp, s, a, s)

include("states.jl")
include("actions.jl")
include("transition.jl")
include("observation.jl")
include("visualization.jl")

end
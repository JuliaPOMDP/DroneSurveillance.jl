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
    QuadCam,
    PerfectCam,
    DroneSurveillanceMDP


const DSPos = SVector{2, Int64}

struct DSState
    quad::DSPos
    agent::DSPos
end

"""
    QuadCam

When used as a camera model, the field of view of the UAV is divided in four region.
If the target is in a corner it is detected perfectly, if it is in the middle of two regions
it is assigned with equal probability to the neighboring regions. If the agent is below the UAV
it is detected with probability 1.
"""
struct QuadCam end 

"""
    PerfectCam

When used as a camera model, the UAV can detect the ground agent with probability 1 when 
it is in its field of view.
"""
struct PerfectCam end

"""
    DroneSurveillanceMDP{M} <: MDP{DSState, Int64, Int64}

# Fields 
- `size::Tuple{Int64, Int64} = (5,5)` size of the grid world
- `region_A::DSPos = [1, 1]` first region to survey, initial state of the quad
- `region_B::DSPos = [size[1], size[2]]` second region to survey
- `fov::Tuple{Int64, Int64} = (3, 3)` size of the field of view of the drone
- `agent_policy::Symbol = :restricted` policy of the other agent
- `camera::M = QuadCam()` observation model, choose between perfect camera and quad camera
- `terminal_state::DSState = DSState([-1, -1], [-1, -1])` a sentinel state to encode terminal states
- `discount_factor::Float64 = 0.95` the discount factor
"""
@with_kw mutable struct DroneSurveillanceMDP{M} <: MDP{DSState, Int64}
    size::Tuple{Int64, Int64} = (5,5)
    region_A::DSPos = [1, 1]
    region_B::DSPos = [size[1], size[2]]
    fov::Tuple{Int64, Int64} = (3, 3)
    agent_policy::Symbol = :restricted
    camera::M = M() # PerfectCam
    terminal_state::DSState = DSState([-1, -1], [-1, -1])
    discount_factor::Float64 = 0.95
end

POMDPs.isterminal(mdp::DroneSurveillanceMDP, s::DSState) = s == mdp.terminal_state 
POMDPs.discount(mdp::DroneSurveillanceMDP) = mdp.discount_factor

function POMDPs.reward(mdp::DroneSurveillanceMDP, s::DSState, a::DSPos)
    if s.quad == s.agent 
        return -100.0
    end
    if s.quad == mdp.region_B
        return 1.0
    end
    return -0.2
end

include("states.jl")
include("actions.jl")
include("transition.jl")
include("visualization.jl")

function __init__()
    # When loaded, set default show backend
    @info "DroneSurveillance loaded!"
end

function stateindex(mdp::DroneSurveillanceMDP{M}, s) where {M}
    s[1] + (s[2]-1) * mdp.size[1]
end
end
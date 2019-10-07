using DroneSurveillance
using Random
using POMDPs
using POMDPModelTools
using POMDPPolicies
using POMDPSimulators
using POMDPTesting
using Test

function test_state_indexing(pomdp::DroneSurveillancePOMDP, ss::Vector{DSState})
    for (i,s) in enumerate(states(pomdp))
        if s != ss[i]
            return false
        end
    end
    return true
end

@testset "state space" begin 
    pomdp = DroneSurveillancePOMDP()
    state_iterator =  states(pomdp)
    ss = ordered_states(pomdp)
    @test length(ss) == length(pomdp)
    @test test_state_indexing(pomdp, ss)
    pomdp = DroneSurveillancePOMDP(size=(7, 10))
    state_iterator =  states(pomdp)
    ss = ordered_states(pomdp)
    @test length(ss) == length(pomdp)
    @test test_state_indexing(pomdp, ss)
end

@testset "action space" begin 
    pomdp = DroneSurveillancePOMDP()
    acts = actions(pomdp)
    @test acts == ordered_actions(pomdp)
    @test length(acts) == length(actions(pomdp))
    @test length(acts) == length(DroneSurveillance.ACTION_DIRS)
end

@testset "transition" begin
    rng = MersenneTwister(1)
    pomdp = DroneSurveillancePOMDP()
    b0 = initialstate_distribution(pomdp)
    @test sum(b0.probs) ≈ 1.0
    s0 = initialstate(pomdp, rng)
    @test s0.quad == DSPos(1, 1)
    @test sum(s0.agent - s0.quad) >= pomdp.fov[1]
    d = transition(pomdp, s0, 1) # move up
    sp = rand(rng, d)
    spp = rand(rng, d)
    @test spp.quad == sp.quad
    @test sp.quad == DSPos(1, 2)
    @test sp.agent != s0.agent
    s = DSState((pomdp.size[1], 1), (4,4))
    d = transition(pomdp, s, 2) # move right
    sp = rand(rng, d)
    @test isterminal(pomdp, sp)
    @test sp == pomdp.terminal_state
    # @inferred transition(pomdp, s0, 3) # not type stable, Union{Deterministic, SparseCat}
    @inferred rand(rng, transition(pomdp, s0, 3))
    trans_prob_consistency_check(pomdp)
end


@testset "observation" begin 
    rng = MersenneTwister(1)
    pomdp = DroneSurveillancePOMDP()
    obs = observations(pomdp)
    @test obs == ordered_observations(pomdp)
    s0 = initialstate(pomdp, rng)
    od = observation(pomdp, 1, s0)
    o = rand(rng, od)
    @test o == 10 # agent should be out
    @inferred observation(pomdp, 6, s0)
    @inferred observation(pomdp, 1, s0)
    s = DSState((2,2), (3,2))
    o = rand(rng, observation(pomdp, 6, s))
    @test o == 2 # east
    s = DSState((2,2), (3,3))
    o = rand(rng, observation(pomdp, 6, s))
    @test o == 6 # north east    
    obs_prob_consistency_check(pomdp)
end

@testset "simulation" begin
    pomdp = DroneSurveillancePOMDP()
    rng = MersenneTwister(1)
    policy = RandomPolicy(pomdp, rng=rng)
    hr = HistoryRecorder(max_steps=10)
    hist = simulate(hr, pomdp, policy)
end

@testset "visualization" begin
    pomdp = DroneSurveillancePOMDP()
    rng = MersenneTwister(1)
    s0 = initialstate(pomdp, rng)
    render(pomdp, Dict(:step=>s0))
end
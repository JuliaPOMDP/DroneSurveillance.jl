
const OBS_DICT = Dict(1 => :N,
                      2 => :E,
                      3 => :S,
                      4 => :W,
                      5 => :DET, # right under the quad
                      6 => :NE,
                      7 => :SE,
                      8 => :SW,
                      9 => :NW,
                      10 => :OUT) # out of the FOV

const OBS_DIRS = SVector(DSPos(0,1),
                        DSPos(1,0),
                        DSPos(0,-1),
                        DSPos(-1,0),
                        DSPos(0,0),
                        DSPos(1,1),
                        DSPos(1,-1),
                        DSPos(-1,-1),
                        DSPos(-1,1))

const OBS_QUAD = [:SW, :NW, :NE, :SE, :DET, :OUT]
const N_OBS_PERFECT = 10
const N_OBS_QUAD = 6

POMDPs.observations(pomdp::DroneSurveillancePOMDP{QuadCam}) = 1:N_OBS_QUAD
POMDPs.observations(pomdp::DroneSurveillancePOMDP{PerfectCam}) = 1:N_OBS_PERFECT
POMDPs.obsindex(pomdp::DroneSurveillancePOMDP, o::Int64) = o

function POMDPs.observation(pomdp::DroneSurveillancePOMDP{QuadCam}, a::Int64, s::DSState)
    obs = SVector{N_OBS_QUAD}(1:N_OBS_QUAD)
    probs = zeros(MVector{N_OBS_QUAD})
    obs_dir = s.agent - s.quad 
    obs_ind = findfirst(isequal(obs_dir), OBS_DIRS)
    if obs_ind == nothing 
        probs[6] = 1.0
        return SparseCat(obs, probs)
    elseif OBS_DICT[obs_ind] == :DET
        probs[5] = 1.0
        return SparseCat(obs, probs)
    elseif OBS_DICT[obs_ind] in [:SW, :NW, :NE, :SE]
        quad_ind = findfirst(isequal(OBS_DICT[obs_ind]), OBS_QUAD)
        probs[quad_ind] = 1.0
        return SparseCat(obs, probs)
    elseif OBS_DICT[obs_ind] == :W
        probs[1] = 0.5
        probs[2] = 0.5
        return SparseCat(obs, probs)
    elseif OBS_DICT[obs_ind] == :N
        probs[2] = 0.5
        probs[3] = 0.5
        return SparseCat(obs, probs)
    elseif OBS_DICT[obs_ind] == :E
        probs[3] = 0.5
        probs[4] = 0.5
        return SparseCat(obs, probs)
    else # OBS_DICT[obs_ind] == :S south
        probs[4] = 0.5
        probs[1] = 0.5
        return SparseCat(obs, probs)
    end
end

function POMDPs.observation(pomdp::DroneSurveillancePOMDP{PerfectCam}, a::Int64, s::DSState)
    obs = SVector{N_OBS_PERFECT}(1:N_OBS_PERFECT)
    probs = zeros(MVector{N_OBS_PERFECT})
    obs_dir = s.agent - s.quad 
    obs_ind = findfirst(isequal(obs_dir), OBS_DIRS)
    if obs_ind == nothing 
        probs[10] = 1.0
        return SparseCat(obs, probs)
    else
        probs[obs_ind] = 1.0
        return SparseCat(obs, probs)
    end
end
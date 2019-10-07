const N_OBS = 10

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



POMDPs.observations(pomdp::DroneSurveillancePOMDP) = 1:N_OBS
POMDPs.obsindex(pomdp::DroneSurveillancePOMDP, o::Int64) = o

function POMDPs.observation(pomdp::DroneSurveillancePOMDP, a::Int64, s::DSState)
    obs = SVector{N_OBS}(1:N_OBS)
    probs = zeros(MVector{N_OBS})
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
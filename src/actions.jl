const N_ACTIONS = 5

const ACTIONS_DICT = Dict(:north => 1, 
                            :east => 2,
                            :south => 3,
                            :west => 4,
                            :hover => 5)

const ACTION_DIRS = (DSPos(0,1),
                     DSPos(1,0),
                     DSPos(0,-1),
                     DSPos(-1,0),
                     DSPos(0,0))
                     
POMDPs.actions(mdp::DroneSurveillanceMDP) = ACTION_DIRS
POMDPs.actionindex(mdp::DroneSurveillanceMDP, idx::Int64) = ACTION_DIRS[idx]

# POMDPs.actions(mdp::DroneSurveillanceMDP, s::DSPos) = 1:N_ACTIONS
POMDPs.actions(mdp::DroneSurveillanceMDP, s::DSPos) = ACTION_DIRS
using Random
using DroneSurveillance
using POMDPs
using POMDPTools
using POMDPGifs # Not included in extras

pomdp = DroneSurveillancePOMDP()

s = DSState(DSPos(2,2), DSPos(4,3))

c = render(pomdp, Dict(:s => s))

policy = RandomPolicy(pomdp)

hr = HistoryRecorder(max_steps=10)
hist = simulate(hr, pomdp, policy)

makegif(pomdp, hist, filename="test.gif", spec="(s,a)")

using SARSOP

solver = SARSOPSolver(precision=1e-3)

policy = solve(solver, pomdp)

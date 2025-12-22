%% Helper function: runTraditionalOptimisation
function [bestCap, bestCost] = runTraditionalOptimisation(w_obj, lb, ub, opts)
% runTraditionalOptimisation  Solve the optimisation problem with a
% deterministic algorithm.
%
% [bestCap, bestCost] = runTraditionalOptimisation(w_obj, lb, ub, opts)
% runs a particle swarm optimisation (PSO) solver to find the optimal
% capacity configuration for the coupled electricity–gas network.  The
% PSO solver used here is MATLAB's built‑in ``particleswarm``.  If you
% have a different solver available (e.g. genetic algorithm or other
% heuristic), you can replace the call accordingly.  The objective
% function wraps the deterministic solver ``iter_couple_most_mpng_24h_merged``.

numVars = numel(lb);
% Define the objective function for PSO
objFun = @(cap) psoObjective(cap, w_obj, opts);
% Set PSO options
psoOpts = optimoptions('particleswarm', 'Display','off', 'SwarmSize', 20, 'MaxIterations', 30);
% Run PSO
[bestCap, bestCost] = particleswarm(objFun, numVars, lb, ub, psoOpts);
end

function f = psoObjective(cap, w_obj, opts)
% psoObjective  Wrapper around the deterministic solver to be minimised.
out = iter_couple_most_mpng_24h_merged(cap, w_obj, opts);
f = out.totalCost;
end
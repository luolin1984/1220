function [initObs, loggedSignals] = capResetFcn()
% capResetFcn.m (v2: CMDP primal-dual state + scenario seed + MDP support)

global ENV

loggedSignals = struct();
loggedSignals.EpisodeStep = 0;
loggedSignals.MaxSteps    = ENV.MaxSteps;

% keep previous action for action-rate penalty / Markovization
loggedSignals.prevAction = (ENV.cap_min + ENV.cap_max) / 2;

% expose opts/config
loggedSignals.opts = ENV;

% primal-dual (Lagrange multipliers) for constraints (CMDP)
% lambda >= 0, updated in capStepFcn
if isfield(ENV,'cmdp') && isfield(ENV.cmdp,'nConstr')
    nC = ENV.cmdp.nConstr;
else
    nC = 2; % default: [voltage_CVaR, gas_risk] constraints
end
loggedSignals.lambda = zeros(nC,1);

% per-episode debug flag
loggedSignals.errPrinted = false;

% scenario seed (optional): helps multi-scenario training reproducibility
% You can pass ENV.iter_opts.seed_base and ENV.iter_opts.seed_span
seed_base = getfield_default(ENV, 'seed_base', 1234);
seed_span = getfield_default(ENV, 'seed_span', 100000);
loggedSignals.scenarioSeed = seed_base + randi(seed_span);

% initial observation:
% v2 uses 5-dim obs: [cost, curt, gas, vdev_CVaR, dA] (all normalized to [0,1])
initObs = zeros(5,1);
end

function v = getfield_default(s, f, d)
if isstruct(s) && isfield(s, f)
    v = s.(f);
else
    v = d;
end
end

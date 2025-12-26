function [nextObs, reward, isDone, loggedSignals] = capStepFcn(action, loggedSignals)
% capStepFcn.m (v3: CMDP primal-dual + CVaR obs + Markovization)

loggedSignals.EpisodeStep = loggedSignals.EpisodeStep + 1;

opts = loggedSignals.opts;

% pass scenarioSeed into opts so evaluate_cap can use it (rng/iter seed)
opts.scenarioSeed = loggedSignals.scenarioSeed;

% defaults
nextObs = ones(5,1);
reward  = getfield_default(opts,'bad_reward',-1.0);
bad     = 1;
loss    = 1.0;
info    = struct('stage',"capStepFcn:init");

% evaluate base metrics + constraint costs
try
    [nextObs, base_reward, bad, loss, info] = evaluate_cap(action, loggedSignals.prevAction, opts);
catch ME
    info.stage   = "capStepFcn:exception";
    info.err_id  = string(ME.identifier);
    info.err_msg = string(ME.message);
    info.err     = getReport(ME,"extended","hyperlinks","off");

    nextObs = ones(5,1);
    base_reward = getfield_default(opts,'bad_reward',-1.0);
    bad = true;
    loss = 1.0;
    info.c = [1;1];
end

% update prevAction
if isfield(info,'action_clip') && isnumeric(info.action_clip)
    loggedSignals.prevAction = info.action_clip(:);
end

% primal-dual update of lambda (Lagrangian CMDP)
% final reward = base_reward - lambda' * c - rho * ||c||^2  (optional quadratic)
cmdp = getfield_default(opts,'cmdp',struct());
eta  = getfield_default(cmdp,'lambda_lr', 0.05);   % dual step size
rho  = getfield_default(cmdp,'quad_rho',  0.00);   % quadratic penalty (optional)
lam_max = getfield_default(cmdp,'lambda_max', 50); % prevent blow-up

c = getfield_default(info,'c',[0;0]);
c = double(c(:));
if ~isfield(loggedSignals,'lambda') || isempty(loggedSignals.lambda)
    loggedSignals.lambda = zeros(numel(c),1);
end
lambda = double(loggedSignals.lambda(:));

% dual ascent: lambda <- max(0, lambda + eta * (c - eps))
% eps allows soft satisfaction (tolerance)
epsC = getfield_default(cmdp,'eps', zeros(numel(c),1));
epsC = double(epsC(:));
if numel(epsC) ~= numel(c), epsC = zeros(numel(c),1); end

lambda = max(0, lambda + eta * (c - epsC));
lambda = min(lambda, lam_max);

loggedSignals.lambda = lambda;
info.lambda = lambda;

% final reward with constraints
reward = double(base_reward) - (lambda.' * c) - rho * sum(c.^2);

% bad handling: still allow lambda update, but you can clamp reward
if bad
    reward = getfield_default(opts,'bad_reward',-1.0);
end

% bookkeeping
loggedSignals.bad  = logical(bad);
loggedSignals.loss = double(loss);
loggedSignals.info = info;

% errPrinted flag per episode
if ~isfield(loggedSignals,'errPrinted') || loggedSignals.EpisodeStep == 1
    loggedSignals.errPrinted = false;
end

% done condition
isDone = logical(loggedSignals.EpisodeStep >= loggedSignals.MaxSteps);

% early terminate on bad (recommended for expensive OPF)
terminate_on_bad = getfield_default(opts,'terminate_on_bad', true);
if terminate_on_bad && bad
    isDone = true;
end

% LIVE print
if getfield_default(opts,'live',true)
    fprintf('[LIVE] step=%d/%d r=%+.4f base=%+.4f bad=%d loss=%.3f obs=[%.3f %.3f %.3f %.3f %.3f] c=[%.3f %.3f] lam=[%.3f %.3f] vdev_cvar=%.4f\n', ...
        loggedSignals.EpisodeStep, loggedSignals.MaxSteps, ...
        double(reward), double(base_reward), logical(bad), double(loss), ...
        nextObs(1), nextObs(2), nextObs(3), nextObs(4), nextObs(5), ...
        safe_get(c,1), safe_get(c,2), safe_get(lambda,1), safe_get(lambda,2), ...
        double(getfield_default(info,'vdev_cvar',NaN)));
end

% print exception details once per episode
if bad && isfield(info,'err') && ~loggedSignals.errPrinted
    loggedSignals.errPrinted = true;
    fprintf('[ERR] stage=%s | id=%s\n', string(getfield_default(info,'stage',"unknown")), string(getfield_default(info,'err_id',"")));
    fprintf('[ERR] msg=%s\n', string(getfield_default(info,'err_msg',"")));
    if isfield(info,'iter_path'), fprintf('[ERR] iter_path=%s\n', string(info.iter_path)); end
    disp(info.err);
end

end

function v = getfield_default(s, f, d)
if isstruct(s) && isfield(s, f)
    v = s.(f);
else
    v = d;
end
end

function x = safe_get(v,i)
try
    if numel(v) >= i, x = double(v(i)); else, x = NaN; end
catch
    x = NaN;
end
end

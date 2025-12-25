function [nextObs, reward, isDone, loggedSignals] = capStepFcn(action, loggedSignals)
% capStepFcn.m (patched: print exception details once per episode)

loggedSignals.EpisodeStep = loggedSignals.EpisodeStep + 1;

opts = loggedSignals.opts;

% 默认输出
nextObs = ones(4,1);
reward  = getfield_default(opts,'bad_reward',-1.0);
bad     = 1;
loss    = 1.0;
info    = struct('stage',"capStepFcn:init");

% evaluate
try
    [nextObs, reward, bad, loss, info] = evaluate_cap(action, loggedSignals.prevAction, opts);
catch ME
    info.stage   = "capStepFcn:exception";
    info.err_id  = string(ME.identifier);
    info.err_msg = string(ME.message);
    info.err     = getReport(ME,"extended","hyperlinks","off");
end

% update memory (guard missing fields when exception happens)
if isfield(info,'action_clip') && isnumeric(info.action_clip)
    loggedSignals.prevAction = info.action_clip;
end
loggedSignals.bad  = bad;
loggedSignals.loss = loss;
loggedSignals.info = info;

% init per-episode flag
if ~isfield(loggedSignals,'errPrinted') || loggedSignals.EpisodeStep == 1
    loggedSignals.errPrinted = false;
end

% stop condition
isDone = logical(loggedSignals.EpisodeStep >= loggedSignals.MaxSteps);

% LIVE print
if getfield_default(opts,'live',true)
    fprintf('[LIVE] step=%d/%d r=%+.4f bad=%d loss=%.3f obs=[%.3f %.3f %.3f %.3f] dA=%.3f vdev_raw=%s vdev_clip=%.4f src=%s\n', ...
        loggedSignals.EpisodeStep, loggedSignals.MaxSteps, reward, bad, loss, ...
        nextObs(1), nextObs(2), nextObs(3), nextObs(4), ...
        getfield_default(info,'dA',NaN), num2str(getfield_default(info,'vdev_raw',NaN)), ...
        getfield_default(info,'vdev_clip',getfield_default(getfield_default(opts,'norm',struct()),'vdev_cap',0.2)), ...
        string(getfield_default(info,'vdev_src',"")));
end

% print exception details (once per episode)
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

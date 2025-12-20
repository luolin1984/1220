function [nextObs, reward, isDone, loggedSignals] = capStepFcn(action, loggedSignals)
% capStepFcn_vdev4.m

loggedSignals.EpisodeStep = loggedSignals.EpisodeStep + 1;

opts = loggedSignals.opts;

% evaluate
[nextObs, reward, bad, loss, info] = evaluate_cap(action, loggedSignals.prevAction, opts);

% update memory
loggedSignals.prevAction = info.action_clip;
loggedSignals.bad        = bad;
loggedSignals.loss       = loss;
loggedSignals.info       = info;

% done?
isDone = loggedSignals.EpisodeStep >= loggedSignals.MaxSteps;

% live log
if getfield_default(opts, 'verbose_live', true)
    fprintf('[LIVE] step=%d/%d  r=%+.4f  bad=%d  loss=%.3f  obs=[%.3f %.3f %.3f %.3f]  dA=%.3f  vdev_raw=%.4f vdev_clip=%.4f src=%s\n', ...
        loggedSignals.EpisodeStep, loggedSignals.MaxSteps, reward, bad, loss, ...
        nextObs(1), nextObs(2), nextObs(3), nextObs(4), ...
        info.dA, info.vdev_raw, info.vdev_clip, string(info.vdev_src));
end
end

function v = getfield_default(s, f, d)
if isstruct(s) && isfield(s, f)
    v = s.(f);
else
    v = d;
end
end

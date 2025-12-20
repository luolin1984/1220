function [initObs, loggedSignals] = capResetFcn()
% capResetFcn_vdev4.m

global ENV

loggedSignals = struct();
loggedSignals.EpisodeStep = 0;
loggedSignals.MaxSteps    = ENV.MaxSteps;

% keep previous action for action-rate penalty
loggedSignals.prevAction = (ENV.cap_min + ENV.cap_max) / 2;

% expose opts/config
loggedSignals.opts = ENV;

% initial observation: neutral (zeros means "good" in our normalized KPIs)
initObs = zeros(4,1);
end

function agent = buildPPOAgent(obsInfo, actInfo, cap_min, cap_max)
% 连续动作 PPO：Gaussian actor + Value critic
[actor, critic] = buildGaussianActorValueCritic(obsInfo, actInfo, cap_min, cap_max);

ppoOpts = rlPPOAgentOptions;
ppoOpts = safeSetOpt(ppoOpts, "SampleTime", 1);
ppoOpts = safeSetOpt(ppoOpts, "DiscountFactor", 0.99);
ppoOpts = safeSetOpt(ppoOpts, "ExperienceHorizon", 128);
ppoOpts = safeSetOpt(ppoOpts, "MiniBatchSize", 256);
ppoOpts = safeSetOpt(ppoOpts, "NumEpoch", 3);
ppoOpts = safeSetOpt(ppoOpts, "ClipFactor", 0.2);
ppoOpts = safeSetOpt(ppoOpts, "EntropyLossWeight", 0.01);
ppoOpts = safeSetOpt(ppoOpts, "GAEFactor", 0.95);

agent = rlPPOAgent(actor, critic, ppoOpts);
end
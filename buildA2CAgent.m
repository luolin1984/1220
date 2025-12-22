function agent = buildA2CAgent(obsInfo, actInfo, cap_min, cap_max)
% 连续动作 A2C：Gaussian actor + Value critic（用 rlACAgent）
[actor, critic] = buildGaussianActorValueCritic(obsInfo, actInfo, cap_min, cap_max);

acOpts = rlACAgentOptions;
acOpts = safeSetOpt(acOpts, "SampleTime", 1);
acOpts = safeSetOpt(acOpts, "DiscountFactor", 0.99);
acOpts = safeSetOpt(acOpts, "EntropyLossWeight", 0.01);
acOpts = safeSetOpt(acOpts, "ExperienceHorizon", 128);

agent = rlACAgent(actor, critic, acOpts);
end
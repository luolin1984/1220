clc; clear; close all;
%% ---------------- 0) 输入 ----------------
global ENV
ENV = struct();

% --- action bounds (12D) ---
% cap_min(4) 电锅炉/热泵额定功率上限（MW），并在构造el_boiler时加 min(el_boiler, cap_in(4))
% cap_min(6) 储能功率或能量上限（MW/MWh），接入 MOST 的 storage 或可调负荷
ENV.cap_min = [  5;  2;  2;   0;  20;   0;   1; 0.06;  1; 0.06;  1; 0.06];
ENV.cap_max = [ 80; 60; 60; 300; 120; 300;   33; 0.85;  33; 0.85;  33; 0.85];

% --- compressor configuration (KEY REQUEST) ---
ENV.comp_ids         = [5 7];
ENV.force_comp_as_el = false;

% --- iter_couple options (forwarded) ---
ENV.iter_opts = struct();

ENV.iter_opts.comp_ids = ENV.comp_ids;
ENV.iter_opts.force_comp_as_el = ENV.force_comp_as_el;

ENV.iter_opts.vdev.eval = 'mpng'; % 可选：MPNG 没有 Vm 时再回退 acopf/acpf（代码里已做）
ENV.iter_opts.vdev.fallback_to_pf = true;

% --- objective weights passed to iter_couple (if used there) ---
ENV.w = [1 1 1 1];

%% ---------------- 1) 归一化/奖励配置（按需可调） ----------------
ENV.norm = struct();
ENV.norm.cost_ref    = 30;    % normalize avg_cost by this (tune)
ENV.norm.curt_ref    = 1.0;   % curtail_ratio already [0,1]
ENV.norm.gas_ref     = 1.0;  % normalize gas_risk by this (tune)
ENV.norm.vdev_cap    = 0.20;  % voltage dev cap (pu)
ENV.norm.dA_lambda   = 0.05;  % action-rate penalty weight (tune)

% --- hard-fail behavior ---
ENV.bad_reward = -1.0;        % only used on hard failures
ENV.hard_fail_on_invalid_kpi = false; % if true, any NaN/Inf KPI => hard fail
ENV.verbose_live = true;
% --- episode horizon ---
ENV.MaxSteps = 5;


lb = [ENV.cap_min(1:6); 1; 1; 1];
ub = [ENV.cap_max(1:6); 33; 33; 33];
ENV.cap_min = lb;
ENV.cap_max = ub;


%% -------- 2) 构造 RL 环境 ----------
obsDim = 4;
actDim = numel(ENV.cap_min);

obsInfo = rlNumericSpec([obsDim 1], "LowerLimit", zeros(obsDim,1), "UpperLimit", ones(obsDim,1));
obsInfo.Name = "obs";

actInfo = rlNumericSpec([actDim 1], "LowerLimit", ENV.cap_min, "UpperLimit", ENV.cap_max);
actInfo.Name = "action";

env = rlFunctionEnv(obsInfo, actInfo, "capStepFcn", "capResetFcn");

%% ---------------- 3) 训练设置（四个算法统一口径） ----------------
cfg = struct();
cfg.MaxEpisodes   = 30;
cfg.MaxSteps      = ENV.MaxSteps;
cfg.EvalEpisodes  = 10;    % 训练后评估回合数
cfg.Seed          = 1;

rng(cfg.Seed);

trainOpts = rlTrainingOptions( ...
    MaxEpisodes = cfg.MaxEpisodes, ...
    MaxStepsPerEpisode = cfg.MaxSteps, ...
    ScoreAveragingWindowLength = min(20, cfg.MaxEpisodes), ...
    StopTrainingCriteria = "EpisodeCount", ...
    StopTrainingValue = cfg.MaxEpisodes, ...
    Verbose = true, ...
    Plots = "training-progress");   % 如果你不想弹窗，改成 "none"

%% ---------------- 4) 构建四个 agent ----------------
agents = struct();

% DDPG
[actorDDPG, criticDDPG] = buildDDPGNetworks(obsInfo, actInfo, ENV.cap_min, ENV.cap_max);
ddpgOpts = rlDDPGAgentOptions(SampleTime=1, DiscountFactor=0.99, TargetSmoothFactor=1e-3, ...
    ExperienceBufferLength=1e6, MiniBatchSize=256);
ddpgOpts.NoiseOptions.Variance = 0.15;        % exploration noise
ddpgOpts.NoiseOptions.VarianceDecayRate = 1e-5;
agents.DDPG = rlDDPGAgent(actorDDPG, criticDDPG, ddpgOpts);

% TD3
[actorTD3, critic1TD3, critic2TD3] = buildTD3Networks(obsInfo, actInfo, ENV.cap_min, ENV.cap_max);
td3Opts = rlTD3AgentOptions(SampleTime=1, DiscountFactor=0.99, TargetSmoothFactor=5e-3, ...
    ExperienceBufferLength=1e6, MiniBatchSize=256);
% TD3 exploration + target-policy smoothing noise
td3Opts.ExplorationModel.Variance = 0.15;
td3Opts.ExplorationModel.VarianceDecayRate = 1e-5;
td3Opts.TargetPolicySmoothModel.Variance = 0.20;
td3Opts.TargetPolicySmoothModel.LowerLimit = -0.5;
td3Opts.TargetPolicySmoothModel.UpperLimit = 0.5;
agents.TD3 = rlTD3Agent(actorTD3, [critic1TD3 critic2TD3], td3Opts);

% --- PPO (continuous Gaussian policy) ---
agents.PPO = buildPPOAgent(obsInfo, actInfo, ENV.cap_min, ENV.cap_max);

% --- A2C (用 AC agent + continuous Gaussian policy) ---
agents.A2C = buildA2CAgent(obsInfo, actInfo, ENV.cap_min, ENV.cap_max);

%% ---------------- 5) 依次训练 + 评估 ----------------
names = {"DDPG","TD3","PPO","A2C"};
results = struct();
results.cfg = cfg;
results.lb = lb;
results.ub = ub;
results.summary = struct();

for i = 1:numel(names)
    name = names{i};
    fprintf('\n================= Training %s =================\n', name);
    agent = agents.(name);

    try
        trainStats = train(agent, env, trainOpts);
        results.(name).trainStats = trainStats;
        results.(name).trainOK = true;
    catch ME
        warning('Training %s failed: %s', name, ME.message);
        results.(name).trainStats = [];
        results.(name).trainOK = false;
    end

    fprintf('----------------- Evaluating %s -----------------\n', name);
    try
        evalRes = evaluateAgentSimple(env, agent, cfg.EvalEpisodes, cfg.MaxSteps);
        results.(name).eval = evalRes;
        results.(name).evalOK = true;
    catch ME
        warning('Evaluation %s failed: %s', name, ME.message);
        results.(name).eval = [];
        results.(name).evalOK = false;
    end
end

%% ---------------- 6) 打印汇总 ----------------
fprintf('\n================= Summary =================\n');
for i = 1:numel(names)
    name = names{i};
    if isfield(results, name) && isfield(results.(name), "eval") && ~isempty(results.(name).eval)
        s = results.(name).eval;
        fprintf('%s | meanReturn=%.4f  std=%.4f  bestReturn=%.4f\n', ...
            name, s.meanReturn, s.stdReturn, s.bestReturn);
    else
        fprintf('%s | (no eval result)\n', name);
    end
end

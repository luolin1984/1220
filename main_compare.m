
clc; clear; close all;
%% ---------------- 0) 输入 ----------------
global ENV
ENV = struct();

% --- action bounds (12D) ---
% cap_min(4) 电锅炉/热泵额定功率上限（MW），并在构造el_boiler时加 min(el_boiler, cap_in(4))
% cap_min(6) 储能功率或能量上限（MW/MWh），接入 MOST 的 storage 或可调负荷
ENV.cap_min = [  5;  2;  2;   1;  20;   1;   1;    1;    1];
ENV.cap_max = [ 80; 60; 60; 300; 120; 300;   33;   33;   33];

% --- compressor configuration ---
ENV.comp_ids         = [5 7];
ENV.force_comp_as_el = true;

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

ENV.norm.cvar_alpha = 0.10;   % CVaR top-10% tail
ENV.constr.vdev_limit = 0.05; % 电压偏差 CVaR 约束阈值（示例，按你系统修）
ENV.constr.gas_limit  = 0.10; % 气网风险阈值（按 gas_risk 定义修）

ENV.cmdp.nConstr   = 2;
ENV.cmdp.lambda_lr = 0.05;    % 对偶步长
ENV.cmdp.lambda_max = 50;
ENV.cmdp.quad_rho  = 0.00;    % 二次惩罚可先关
ENV.cmdp.eps = [0;0];         % 允许的软容忍

ENV.terminate_on_bad = true;

% 多场景随机性（可选）
ENV.seed_base = 1234;
ENV.seed_span = 100000;

% --- hard-fail behavior ---
ENV.bad_reward = -1.0;        % only used on hard failures
ENV.hard_fail_on_invalid_kpi = false; % if true, any NaN/Inf KPI => hard fail
ENV.verbose_live = true;
% --- episode horizon ---
ENV.MaxSteps = 5;

%% ---------------- 2) 构造 RL 环境 ----------
obsDim = 5;
actDim = numel(ENV.cap_min);

obsInfo = rlNumericSpec([obsDim 1], "LowerLimit", zeros(obsDim,1), "UpperLimit", ones(obsDim,1));
obsInfo.Name = "obs";

actInfo = rlNumericSpec([actDim 1], "LowerLimit", ENV.cap_min, "UpperLimit", ENV.cap_max);
actInfo.Name = "action";

env = rlFunctionEnv(obsInfo, actInfo, "capStepFcn", "capResetFcn");

%% ---------------- 3) 训练设置（四个算法统一口径） ----------------
cfg = struct();
cfg.MaxEpisodes   = 10; % 30
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
ddpgOpts.NoiseOptions.Variance = 0.15;
ddpgOpts.NoiseOptions.VarianceDecayRate = 1e-5;
agents.DDPG = rlDDPGAgent(actorDDPG, criticDDPG, ddpgOpts);

% TD3
[actorTD3, critic1TD3, critic2TD3] = buildTD3Networks(obsInfo, actInfo, ENV.cap_min, ENV.cap_max);
td3Opts = rlTD3AgentOptions(SampleTime=1, DiscountFactor=0.99, TargetSmoothFactor=5e-3, ...
    ExperienceBufferLength=1e6, MiniBatchSize=256);
td3Opts.ExplorationModel.Variance = 0.15;
td3Opts.ExplorationModel.VarianceDecayRate = 1e-5;
td3Opts.TargetPolicySmoothModel.Variance = 0.20;
td3Opts.TargetPolicySmoothModel.LowerLimit = -0.5;
td3Opts.TargetPolicySmoothModel.UpperLimit = 0.5;
agents.TD3 = rlTD3Agent(actorTD3, [critic1TD3 critic2TD3], td3Opts);

% PPO / A2C
agents.PPO = buildPPOAgent(obsInfo, actInfo, ENV.cap_min, ENV.cap_max);
agents.A2C = buildA2CAgent(obsInfo, actInfo, ENV.cap_min, ENV.cap_max);

%% ---------------- 5) 依次训练 + 评估 ----------------
names = {"DDPG","TD3","PPO","A2C"};

results = struct();
results.cfg = cfg;
results.lb  = ENV.cap_min;     % 修复：原来 lb/ub 未定义
results.ub  = ENV.cap_max;
results.summary = struct();

for i = 1:numel(names)
    name = names{i};
    fprintf('\n================= Training %s =================\n', name);
    agent = agents.(name);

    % ✅ 关键：用 safe_train 兼容不同 MATLAB 版本 train 输出形式
    [trainedAgent, trainStats, trainErr] = safe_train(agent, env, trainOpts);

    if isempty(trainErr)
        agent = trainedAgent;          % 对“2输出版本”是新的 agent
        agents.(name) = agent;         % 对“1输出版本”agent 通常已在原地更新，这句也安全
        results.(name).trainStats = trainStats;
        results.(name).trainOK = true;

        % 额外：打印一下曲线点数，确保不是空
        try
            [ep_dbg, r_dbg] = local_extract_reward(trainStats);
            fprintf('[%s] trainStats episodes=%d\n', name, numel(r_dbg));
        catch
        end
    else
        warning('Training %s failed:\n%s', name, getReport(trainErr,'extended','hyperlinks','off'));
        results.(name).trainStats = [];
        results.(name).trainOK = false;
        results.(name).trainErr = trainErr;
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

%% ---------------- 5.5) 其他对比方法（非 RL Baselines） ----------------
baseline = struct();

% 评估口径：与 evaluateAgentSimple 一致（每回合 Return = sum(reward)）
baseline.EvalEpisodes = cfg.EvalEpisodes;
baseline.MaxSteps     = cfg.MaxSteps;

% 为了让黑箱搜索别太慢：搜索阶段用更少回合做近似评估，最终再用 EvalEpisodes 复评
baseline.SearchEvalEpisodes = 2;

% 随机搜索次数（按你 iter_couple 的耗时调）
baseline.RandomN = 120;

lb = ENV.cap_min(:);
ub = ENV.cap_max(:);

% --- 1) 固定策略（非常必要的常识基线） ---
baseCaps = struct();
baseCaps.CAP_MIN = lb;
baseCaps.CAP_MID = (lb + ub)/2;
baseCaps.CAP_MAX = ub;

% --- 2) Random Search（黑箱强基线） ---
[cap_randbest, randHist] = baseline_random_search(env, lb, ub, baseline.RandomN, ...
    baseline.SearchEvalEpisodes, baseline.MaxSteps, cfg.Seed+1000);
baseCaps.RAND = cap_randbest;

% --- 3) GA / PSO / fmincon（有工具箱就用，没有就自动跳过） ---
baseCaps.GA      = baseline_try_ga(env, lb, ub, baseCaps.CAP_MID, baseline, cfg.Seed+2000);
baseCaps.PSO     = baseline_try_pso(env, lb, ub, baseCaps.CAP_MID, baseline, cfg.Seed+3000);
baseCaps.FMINCON = baseline_try_fmincon(env, lb, ub, baseCaps.CAP_MID, baseline, cfg.Seed+4000);

% --- 4) 统一复评（EvalEpisodes）并写入 results ---
bNames = fieldnames(baseCaps);
results.baseline = struct();

fprintf('\n================= Evaluating Baselines =================\n');
for k = 1:numel(bNames)
    bn  = bNames{k};
    cap = baseCaps.(bn);

    try
        s = evaluateFixedCapPolicy(env, cap, baseline.EvalEpisodes, baseline.MaxSteps, cfg.Seed+5000+k);
        results.baseline.(bn).cap  = cap;
        results.baseline.(bn).eval = s;
        fprintf('%s | meanReturn=%.4f  std=%.4f  bestReturn=%.4f\n', ...
            bn, s.meanReturn, s.stdReturn, s.bestReturn);
    catch ME
        warning('Baseline %s eval failed: %s', bn, ME.message);
        results.baseline.(bn).cap  = cap;
        results.baseline.(bn).eval = [];
    end
end

% 可选：保存 baseline 搜索过程
try
    save('baseline_search.mat', 'baseCaps', 'randHist', 'baseline');
catch
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
% ---- Baseline summary ----
if isfield(results,'baseline')
    bNames = fieldnames(results.baseline);
    for k = 1:numel(bNames)
        bn = bNames{k};
        ev = results.baseline.(bn).eval;
        if ~isempty(ev)
            fprintf('BASE-%s | meanReturn=%.4f  std=%.4f  bestReturn=%.4f\n', ...
                bn, ev.meanReturn, ev.stdReturn, ev.bestReturn);
        else
            fprintf('BASE-%s | (no eval result)\n', bn);
        end
    end
end


%% ---------------- 7) 训练曲线对比（四算法叠加） ----------------
try
    figure('Name','Training Curves (4 RL Methods)','Color','w');
    hold on; grid on;

    anyPlotted = false;

    for i = 1:numel(names)
        name = names{i};
        if ~isfield(results, name) || ~isfield(results.(name), 'trainStats') || isempty(results.(name).trainStats)
            continue;
        end

        ts = results.(name).trainStats;
        [ep, r, rAvg] = local_extract_reward(ts);

        if isempty(ep) || isempty(r), continue; end

        plot(ep, r, 'DisplayName', sprintf('%s: EpisodeReward', name));
        anyPlotted = true;
        if ~isempty(rAvg)
            plot(ep, rAvg, '--', 'DisplayName', sprintf('%s: AverageReward', name));
        end
    end

    xlabel('Episode');
    ylabel('Reward');
    title('Training Curves Comparison');

    if anyPlotted
        legend('Location','best');
    else
        title('Training Curves Comparison (NO DATA PLOTTED)');
        text(0.05,0.9,'No trainStats available. Check results.(agent).trainErr', 'Units','normalized');
    end
catch ME
    warning('[plot] training curves failed: %s', ME.message);
end

%% ---------- local helper: robust reward extraction ----------
function [ep, r, rAvg] = local_extract_reward(ts)
ep = []; r = []; rAvg = [];

% table
if istable(ts)
    vn = ts.Properties.VariableNames;
    if any(strcmp(vn,'EpisodeIndex')),  ep   = ts.EpisodeIndex;  end
    if any(strcmp(vn,'EpisodeReward')), r    = ts.EpisodeReward; end
    if any(strcmp(vn,'AverageReward')), rAvg = ts.AverageReward; end
    if isempty(ep) && ~isempty(r), ep = (1:numel(r)).'; end
    return;
end

% struct
if isstruct(ts)
    if isfield(ts,'EpisodeIndex'),  ep   = ts.EpisodeIndex;  end
    if isfield(ts,'EpisodeReward'), r    = ts.EpisodeReward; end
    if isfield(ts,'AverageReward'), rAvg = ts.AverageReward; end
    if isempty(ep) && ~isempty(r), ep = (1:numel(r)).'; end
    return;
end

% object (just in case)
try
    if isprop(ts,'EpisodeIndex'),  ep   = ts.EpisodeIndex;  end
    if isprop(ts,'EpisodeReward'), r    = ts.EpisodeReward; end
    if isprop(ts,'AverageReward'), rAvg = ts.AverageReward; end
    if isempty(ep) && ~isempty(r), ep = (1:numel(r)).'; end
catch
end
end


function [agentOut, stats, trainErr] = safe_train(agentIn, env, trainOpts)
%SAFE_TRAIN 兼容不同 MATLAB RL Toolbox 版本的 train 输出形式
% - 有些版本: [agentOut, stats] = train(...)
% - 有些版本: stats = train(...), agent 在原地更新

trainErr = [];
stats    = [];
agentOut = agentIn;

try
    % 先尝试“2输出”写法（若你的版本支持，这是最理想的）
    [agentOut, stats] = train(agentIn, env, trainOpts);
    return;
catch ME
    if strcmp(ME.identifier, "MATLAB:TooManyOutputs")
        % 回退到“1输出”写法
        try
            stats    = train(agentIn, env, trainOpts);
            agentOut = agentIn;  % 训练后 agentIn 通常已被更新（句柄对象）
            trainErr = [];
            return;
        catch ME2
            trainErr = ME2;
            return;
        end
    else
        trainErr = ME;
        return;
    end
end
end

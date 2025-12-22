clc; clear; close all;
%% -------- Global env config (shared by reset/step/evaluate) ----------
global ENV
ENV = struct();

% --- action bounds (12D) ---
ENV.cap_min = [  5;  2;  2;   0;  20;   0;   1; 0.06;  1; 0.06;  1; 0.06];
ENV.cap_max = [ 80; 60; 60; 300; 120; 300;   33; 0.85;  33; 0.85;  33; 0.85];

% --- reward shaping / normalization refs ---
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

% --- compressor configuration (KEY REQUEST) ---
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

% --- episode horizon ---
ENV.MaxSteps = 5;

%%
w_obj = ENV.w;
lb = [ENV.cap_min(1:6); 1];
ub = [ENV.cap_max(1:6); 33];

% Options structure controlling the solver and simulation.  The fields
% used by the underlying solver ``iter_couple_most_mpng_24h_merged``
% include verbosity levels, solver tolerances and any other
% problem‑specific flags.  See the corresponding solver implementation
% for more details.  Feel free to customise these fields as needed.
opts = struct();
opts.verbose = false;    % suppress verbose solver output
opts.maxIterations = 50; % maximum iterations for the nested solver

%% Define the list of RL algorithms to compare
algorithms = {'DDPG','TD3','PPO','A2C'};

% Preallocate structure to hold results
results = struct();

% Loop over all algorithms and perform training and evaluation
for k = 1:numel(algorithms)
    algName = algorithms{k};
    fprintf('=== Training agent using %s algorithm ===\n', algName);
    try
        [agentCost, agentOut, agentObjVals] = trainAndEvaluateAgent(algName, w_obj, lb, ub, opts);
        results.(algName).cost = agentCost;
        results.(algName).output = agentOut;
        results.(algName).objVals = agentObjVals;
        fprintf('Finished training %s agent. Total cost: %.4f\n', algName, agentCost);
    catch ME
        % Catch any errors during training or evaluation and store them
        warning('An error occurred while training the %s agent: %s', algName, ME.message);
        results.(algName).error = ME;
    end
end

%% Traditional optimisation benchmark (e.g. PSO or deterministic solver)
fprintf('=== Running traditional optimisation (PSO) ===\n');
try
    [capOpt, psoCost] = runTraditionalOptimisation(w_obj, lb, ub, opts);
    results.PSO.capacity = capOpt;
    results.PSO.cost = psoCost;
    fprintf('Finished PSO optimisation. Total cost: %.4f\n', psoCost);
catch ME
    warning('An error occurred while running the traditional optimiser: %s', ME.message);
    results.PSO.error = ME;
end

%% Display summary of results
disp('=== Summary of algorithm performance ===');
algFields = fieldnames(results);
for i = 1:numel(algFields)
    fname = algFields{i};
    if isfield(results.(fname),'cost')
        fprintf('%s: Cost = %.4f\n', fname, results.(fname).cost);
    else
        fprintf('%s: Error encountered\n', fname);
    end
end


% %% -------- RL environment specs ----------
% obsDim = 4;
% actDim = numel(ENV.cap_min);
% 
% obsInfo = rlNumericSpec([obsDim 1], "LowerLimit", zeros(obsDim,1), "UpperLimit", ones(obsDim,1));
% obsInfo.Name = "obs";
% 
% actInfo = rlNumericSpec([actDim 1], "LowerLimit", ENV.cap_min, "UpperLimit", ENV.cap_max);
% actInfo.Name = "action";
% 
% env = rlFunctionEnv(obsInfo, actInfo, "capStepFcn", "capResetFcn");
% 
% %% -------- Build agents (DDPG + TD3) ----------
% agents = struct();
% 
% % DDPG
% [actorDDPG, criticDDPG] = buildDDPGNetworks(obsInfo, actInfo, ENV.cap_min, ENV.cap_max);
% ddpgOpts = rlDDPGAgentOptions( ...
%     SampleTime=1, ...
%     DiscountFactor=0.99, ...
%     TargetSmoothFactor=1e-3, ...
%     ExperienceBufferLength=1e6, ...
%     MiniBatchSize=256);
% ddpgOpts.NoiseOptions.Variance = 0.15;        % exploration noise
% ddpgOpts.NoiseOptions.VarianceDecayRate = 1e-5;
% agents.DDPG = rlDDPGAgent(actorDDPG, criticDDPG, ddpgOpts);
% 
% % TD3
% [actorTD3, critic1TD3, critic2TD3] = buildTD3Networks(obsInfo, actInfo, ENV.cap_min, ENV.cap_max);
% td3Opts = rlTD3AgentOptions( ...
%     SampleTime=1, ...
%     DiscountFactor=0.99, ...
%     TargetSmoothFactor=5e-3, ...
%     ExperienceBufferLength=1e6, ...
%     MiniBatchSize=256);
% % TD3 exploration + target-policy smoothing noise
% td3Opts.ExplorationModel.Variance = 0.15;
% td3Opts.ExplorationModel.VarianceDecayRate = 1e-5;
% td3Opts.TargetPolicySmoothModel.Variance = 0.20;
% td3Opts.TargetPolicySmoothModel.LowerLimit = -0.5;
% td3Opts.TargetPolicySmoothModel.UpperLimit = 0.5;
% agents.TD3 = rlTD3Agent(actorTD3, [critic1TD3 critic2TD3], td3Opts);
% 
% %% -------- Choose agent to train ----------
% cfg = struct();
% cfg.train_agent = "TD3";  % "TD3" | "DDPG" | "BOTH"
% cfg.maxEpisodes = 30;
% cfg.maxStepsPerEpisode = ENV.MaxSteps;
% 
% trainOpts = rlTrainingOptions( ...
%     MaxEpisodes=cfg.maxEpisodes, ...
%     MaxStepsPerEpisode=cfg.maxStepsPerEpisode, ...
%     ScoreAveragingWindowLength=20, ...
%     StopTrainingCriteria="EpisodeCount", ...
%     StopTrainingValue=cfg.maxEpisodes, ...
%     Verbose=true, ...
%     Plots="training-progress");
% 
% stats = struct();
% switch upper(string(cfg.train_agent))
%     case "DDPG"
%         stats.DDPG = train(agents.DDPG, env, trainOpts);
%     case "TD3"
%         stats.TD3  = train(agents.TD3, env, trainOpts);
%     case "BOTH"
%         stats.DDPG = train(agents.DDPG, env, trainOpts);
%         stats.TD3  = train(agents.TD3, env, trainOpts);
%     otherwise
%         error("Unknown cfg.train_agent = %s", cfg.train_agent);
% end
% 
% out = struct("env", env, "agents", agents, "stats", stats, "cfg", cfg);
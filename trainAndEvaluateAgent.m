function [trainedAgent, performance] = trainAndEvaluateAgent(algorithm, obsInfo, actInfo, w_obj, envOpts, numEpisodes, maxSteps)
% trainAndEvaluateAgent  Trains and evaluates a reinforcement learning agent.
%
%   [trainedAgent, performance] = trainAndEvaluateAgent(ALGORITHM, obsInfo, actInfo,
%   w_obj, envOpts, numEpisodes, maxSteps) creates a custom rlFunctionEnv for
%   the power–gas coupled capacity sizing problem, trains the specified
%   reinforcement learning agent and returns the trained agent along with
%   basic performance statistics.  The reset function of the custom
%   environment is created as a closure and therefore accepts no input
%   arguments, as required by the RL toolbox.  All additional inputs such
%   as the objective weight vector w_obj or environment options envOpts are
%   captured in the closure.  The step function also captures w_obj.

% Validate inputs
if nargin < 7 || isempty(maxSteps)
    maxSteps = 1; %#ok<AGROW> default to one step if not specified
end
if nargin < 6 || isempty(numEpisodes)
    numEpisodes = 10; %#ok<AGROW> default number of training episodes
end

% Create the environment.  The reset and step functions are defined as
% anonymous functions so that they have zero input arguments.  The
% environment options and weight vector are captured from the workspace.
resetFcn = @() capResetWrapper(envOpts.cap_init, envOpts);
stepFcn  = @(action) capStepWrapper(action, w_obj);

% Create an rlFunctionEnv using the observation and action specifications
env = rlFunctionEnv(obsInfo, actInfo, stepFcn, resetFcn);

% Select the appropriate agent and configuration based on the requested
% algorithm.  All agents are created using the Environment's
% observation/action info and reasonable default hyper‑parameters.  These
% settings can be tuned as required.
switch lower(algorithm)
    case 'ddpg'
        % DDPG requires deterministic actor and critic networks
        actorNetwork  = rlRepresentation(getDDPGActorNetwork(obsInfo, actInfo), obsInfo, actInfo);
        criticNetwork = rlRepresentation(getDDPGCriticNetwork(obsInfo, actInfo), obsInfo, actInfo);
        agentOpts = rlDDPGAgentOptions('SampleTime',1);
        agent = rlDDPGAgent(actorNetwork, criticNetwork, agentOpts);

    case 'td3'
        % TD3 is built on twin critics and deterministic policy
        actorNetwork  = rlRepresentation(getTD3ActorNetwork(obsInfo, actInfo), obsInfo, actInfo);
        criticNetwork1 = rlRepresentation(getTD3CriticNetwork(obsInfo, actInfo), obsInfo, actInfo);
        criticNetwork2 = rlRepresentation(getTD3CriticNetwork(obsInfo, actInfo), obsInfo, actInfo);
        agentOpts = rlTD3AgentOptions('SampleTime',1);
        agent = rlTD3Agent(actorNetwork, [criticNetwork1 criticNetwork2], agentOpts);

    case 'ppo'
        % Proximal Policy Optimisation (actor‑critic)
        actorNetwork  = rlRepresentation(getPPOActorNetwork(obsInfo, actInfo), obsInfo, actInfo);
        criticNetwork = rlRepresentation(getPPOCriticNetwork(obsInfo, actInfo), obsInfo, actInfo);
        agentOpts = rlPPOAgentOptions('SampleTime',1);
        agent = rlPPOAgent(actorNetwork, criticNetwork, agentOpts);

    case 'a2c'
        % Advantage Actor Critic
        actorNetwork  = rlRepresentation(getA2CActorNetwork(obsInfo, actInfo), obsInfo, actInfo);
        criticNetwork = rlRepresentation(getA2CCriticNetwork(obsInfo, actInfo), obsInfo, actInfo);
        agentOpts = rlACAgentOptions('SampleTime',1);
        agent = rlACAgent(actorNetwork, criticNetwork, agentOpts);

    otherwise
        error('Unsupported algorithm "%s".  Choose from ''ddpg'', ''td3'', ''ppo'' or ''a2c''.', algorithm);
end

% Configure training options.  Use parallel training by default if
% available; adjust criteria such as MaxEpisodes or TargetAvgReward as
% needed.  The StopOnError flag is enabled to abort training when
% encountering an error.
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',    numEpisodes,...
    'MaxStepsPerEpisode', maxSteps,...
    'ScoreAveragingWindowLength', 5,...
    'StopTrainingCriteria', 'None',...
    'StopTrainingValue',   NaN,...
    'UseParallel',         false,...
    'StopOnError',         'on');

% Train the agent.  Catch and rethrow any errors so that the calling
% script can handle failures cleanly.
try
    trainingInfo = train(agent, env, trainOpts);
catch trainErr
    error('Training failed for algorithm %s: %s', algorithm, trainErr.message);
end

trainedAgent = agent;

% Evaluate the trained policy.  Simulate a single episode and collect
% performance metrics.  The returned performance struct contains the
% cumulative reward and the final capacity decisions.
try
    simOpts = rlSimulationOptions('MaxSteps', maxSteps);
    experience  = sim(env, trainedAgent, simOpts);
    performance.TotalReward = sum([experience.Reward]);
    performance.Observation = experience.Observation{end};
catch simErr
    warning('Simulation failed for algorithm %s: %s', algorithm, simErr.message);
    performance = struct('TotalReward', NaN, 'Observation', []);
end

end

%% Helper functions for creating minimal neural networks
function actorNet = getDDPGActorNetwork(obsInfo, actInfo)
% create a simple fully connected actor network for DDPG
obsDims = prod(obsInfo.Dimension);
actDims = prod(actInfo.Dimension);
actorNet = layerGraph();
actorNet = addLayers(actorNet, featureInputLayer(obsDims));
actorNet = addLayers(actorNet, fullyConnectedLayer(64));
actorNet = addLayers(actorNet, reluLayer());
actorNet = addLayers(actorNet, fullyConnectedLayer(64));
actorNet = addLayers(actorNet, reluLayer());
actorNet = addLayers(actorNet, fullyConnectedLayer(actDims));
actorNet = addLayers(actorNet, tanhLayer());
actorNet = connectLayers(actorNet, 'featureinput', 'fullyconnected');
actorNet = connectLayers(actorNet, 'fullyconnected', 'relu');
actorNet = connectLayers(actorNet, 'relu', 'fullyconnected_1');
actorNet = connectLayers(actorNet, 'fullyconnected_1', 'relu_1');
actorNet = connectLayers(actorNet, 'relu_1', 'fullyconnected_2');
actorNet = connectLayers(actorNet, 'fullyconnected_2', 'tanh');
end

function criticNet = getDDPGCriticNetwork(obsInfo, actInfo)
% create a simple critic network for DDPG
obsDims = prod(obsInfo.Dimension);
actDims = prod(actInfo.Dimension);
statePath = featureInputLayer(obsDims);
actionPath = featureInputLayer(actDims);
stateLayer = fullyConnectedLayer(64);
actionLayer = fullyConnectedLayer(64);
commonLayer = additionLayer(2);
commonFC1 = fullyConnectedLayer(64);
commonRelu1 = reluLayer();
valueLayer = fullyConnectedLayer(1);
criticNet = layerGraph(statePath);
criticNet = addLayers(criticNet, actionPath);
criticNet = addLayers(criticNet, stateLayer);
criticNet = addLayers(criticNet, actionLayer);
criticNet = addLayers(criticNet, commonLayer);
criticNet = addLayers(criticNet, commonFC1);
criticNet = addLayers(criticNet, commonRelu1);
criticNet = addLayers(criticNet, valueLayer);
criticNet = connectLayers(criticNet, 'featureinput', 'fullyconnected');
criticNet = connectLayers(criticNet, 'featureinput_1', 'fullyconnected_1');
criticNet = connectLayers(criticNet, 'fullyconnected', 'addition/in1');
criticNet = connectLayers(criticNet, 'fullyconnected_1', 'addition/in2');
criticNet = connectLayers(criticNet, 'addition', 'fullyconnected_2');
criticNet = connectLayers(criticNet, 'fullyconnected_2', 'relu');
criticNet = connectLayers(criticNet, 'relu', 'fullyconnected_3');
end

function actorNet = getTD3ActorNetwork(obsInfo, actInfo)
% identical to DDPG actor for brevity
actorNet = getDDPGActorNetwork(obsInfo, actInfo);
end

function criticNet = getTD3CriticNetwork(obsInfo, actInfo)
% identical to DDPG critic for brevity
criticNet = getDDPGCriticNetwork(obsInfo, actInfo);
end

function actorNet = getPPOActorNetwork(obsInfo, actInfo)
% create a simple actor for PPO
obsDims = prod(obsInfo.Dimension);
actDims = prod(actInfo.Dimension);
actorNet = layerGraph();
actorNet = addLayers(actorNet, featureInputLayer(obsDims));
actorNet = addLayers(actorNet, fullyConnectedLayer(64));
actorNet = addLayers(actorNet, reluLayer());
actorNet = addLayers(actorNet, fullyConnectedLayer(64));
actorNet = addLayers(actorNet, reluLayer());
actorNet = addLayers(actorNet, fullyConnectedLayer(actDims));
actorNet = addLayers(actorNet, softmaxLayer());
actorNet = connectLayers(actorNet, 'featureinput', 'fullyconnected');
actorNet = connectLayers(actorNet, 'fullyconnected', 'relu');
actorNet = connectLayers(actorNet, 'relu', 'fullyconnected_1');
actorNet = connectLayers(actorNet, 'fullyconnected_1', 'relu_1');
actorNet = connectLayers(actorNet, 'relu_1', 'fullyconnected_2');
actorNet = connectLayers(actorNet, 'fullyconnected_2', 'softmax');
end

function criticNet = getPPOCriticNetwork(obsInfo, actInfo)
% PPO critic similar to DDPG critic
criticNet = getDDPGCriticNetwork(obsInfo, actInfo);
end

function actorNet = getA2CActorNetwork(obsInfo, actInfo)
% create a simple actor for A2C
actorNet = getPPOActorNetwork(obsInfo, actInfo);
end

function criticNet = getA2CCriticNetwork(obsInfo, actInfo)
% A2C critic similar to PPO critic
criticNet = getPPOCriticNetwork(obsInfo, actInfo);
end

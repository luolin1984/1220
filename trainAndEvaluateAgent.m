%% Helper function: trainAndEvaluateAgent
function [totalCost, out, objVals] = trainAndEvaluateAgent(algName, w_obj, lb, ub, opts)
% trainAndEvaluateAgent  Train a reinforcement learning agent and evaluate it.
%
% [totalCost, out, objVals] = trainAndEvaluateAgent(algName, w_obj, lb, ub, opts)
% trains an agent of type specified by ``algName`` on the electricity–
% gas network optimisation problem.  After training the policy is
% evaluated by mapping the learned action back into the capacity domain
% and running the deterministic solver ``iter_couple_most_mpng_24h_merged``.
%
% Inputs:
%   algName - String specifying the RL algorithm to use ('DDPG', 'TD3',
%             'PPO' or 'A2C').  The comparison is case-insensitive.
%   w_obj   - Weight vector for the multi‑objective cost function.
%   lb, ub  - Vectors defining the lower and upper bounds of the
%             decision variables (capacities).
%   opts    - Structure of additional options passed through to the
%             deterministic solver.
%
% Outputs:
%   totalCost - Scalar total cost of the capacity solution found by the
%               trained agent (sum of weighted objectives).
%   out       - Structure returned by ``iter_couple_most_mpng_24h_merged``
%               containing detailed simulation results for the selected
%               capacity.
%   objVals   - Vector of individual objective values (prior to
%               weighting) returned by the solver.

%% Create RL environment wrapper
% Determine the dimensionality of the action space from bounds
obsDim = 4;
actionDims = numel(lb);

% Since the state vector returned by the environment is unknown at this
% stage we use a generic dimension of 1 for the observation space.  The
% RL function environment will automatically adapt at runtime when the
% wrapper functions return actual observations.  The action space is
% continuous and bounded in [-1,1] for each dimension; the bounds are
% later used to scale actions into the capacity domain.
obsInfo = rlNumericSpec([obsDim 1], "LowerLimit", zeros(obsDim,1), "UpperLimit", ones(obsDim,1));
obsInfo.Name = 'State';
actInfo = rlNumericSpec([actionDims 1], 'LowerLimit', lb, 'UpperLimit', ub);
actInfo.Name = 'CapacityAction';

% Instantiate the RL function environment with the wrapper functions
env = rlFunctionEnv(obsInfo, actInfo, @capResetWrapper, @capStepWrapper);

%% Design actor and critic networks depending on the algorithm
[actor, critic] = createActorCritic(algName, obsInfo, actInfo);

%% Select agent and options
switch upper(algName)
    case 'DDPG'
        agentOpts = rlDDPGAgentOptions('SampleTime',1);
        agentOpts.DiscountFactor = 0.99;
        agentOpts.MiniBatchSize = 128;
        agentOpts.NoiseOptions.StandardDeviation = 0.2;
        agentOpts.NoiseOptions.StandardDeviationDecayRate = 1e-5;
        agentOpts.CriticOptimizerOptions.LearnRate = 1e-3;
        agentOpts.ActorOptimizerOptions.LearnRate = 1e-4;
        agent = rlDDPGAgent(actor, critic, agentOpts);
    case 'TD3'
        agentOpts = rlTD3AgentOptions('SampleTime',1);
        agentOpts.DiscountFactor = 0.99;
        agentOpts.TargetUpdateFrequency = 2;
        agentOpts.MiniBatchSize = 128;
        agentOpts.NoiseOptions.StandardDeviation = 0.2;
        agentOpts.NoiseOptions.StandardDeviationDecayRate = 1e-5;
        agentOpts.CriticOptimizerOptions.LearnRate = 1e-3;
        agentOpts.ActorOptimizerOptions.LearnRate = 1e-4;
        agent = rlTD3Agent(actor, critic, agentOpts);
    case 'PPO'
        agentOpts = rlPPOAgentOptions('SampleTime',1);
        agentOpts.DiscountFactor = 0.99;
        agentOpts.ClipFactor = 0.2;
        agentOpts.EntropyLossWeight = 0.01;
        agentOpts.MiniBatchSize = 64;
        agentOpts.ExperienceHorizon = 2048;
        agentOpts.LearnRate = 5e-4;
        agent = rlPPOAgent(actor, critic, agentOpts);
    case 'A2C'
        agentOpts = rlACAgentOptions('SampleTime',1);
        agentOpts.DiscountFactor = 0.99;
        agentOpts.EntropyLossWeight = 0.01;
        agentOpts.CriticOptimizerOptions.LearnRate = 1e-3;
        agentOpts.ActorOptimizerOptions.LearnRate = 1e-4;
        agent = rlACAgent(actor, critic, agentOpts);
    otherwise
        error('Unsupported algorithm: %s', algName);
end

%% Training options
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',50,...
    'MaxStepsPerEpisode',200,...
    'ScoreAveragingWindowLength',5,...
    'Verbose',false,...
    'Plots','training-progress');

% You may want to stop training early if a certain threshold is met; set
% the following field accordingly.  Here we leave it empty.
trainOpts.StopTrainingCriteria = '';
trainOpts.StopTrainingValue = inf;

% Train the agent
trainingStats = train(agent, env, trainOpts);

%% Evaluation
% Evaluate the trained policy by computing the optimal capacity from the
% actor, scaling it into the physical bounds and running the coupled
% solver.  We assume that the environment state at time zero is not
% important for determining the capacities; hence we use a dummy zero
% observation for computing the deterministic action.
dummyState = zeros(obsInfo.Dimension);

% Obtain the action from the actor
deterministicAction = getAction(getActor(agent), dummyState);

% Rescale the action into the capacity domain.  Actions lie in [-1,1],
% so we map them linearly into [lb, ub].
capacity = 0.5*(ub + lb) + 0.5*(ub - lb).*deterministicAction;

% Run the deterministic coupled network solver with the learned capacity
% to obtain the cost and individual objective values.  The function
% ``iter_couple_most_mpng_24h_merged`` must be on the path.
out = iter_couple_most_mpng_24h_merged(capacity, w_obj, opts);
totalCost = out.totalCost;
objVals = out.objVals;

end

%% Helper function: createActorCritic
function [actor, critic] = createActorCritic(algName, obsInfo, actInfo)
% createActorCritic  Build actor and critic networks for the specified algorithm.
%
% The architecture is chosen to be simple yet expressive enough for
% continuous control.  You may experiment with different network sizes
% and activation functions to improve performance.

numObs = prod(obsInfo.Dimension);
numAct = prod(actInfo.Dimension);

% Shared feature extraction network for the state input
statePath = [ ...
    featureInputLayer(numObs,'Normalization','none','Name','state')
    fullyConnectedLayer(64,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(64,'Name','fc2')
    reluLayer('Name','relu2') ];

switch upper(algName)
    case {'DDPG','TD3'}
        % Actor network (deterministic)
        actorNetwork = [ statePath
            fullyConnectedLayer(32,'Name','fcAct1')
            reluLayer('Name','reluAct1')
            fullyConnectedLayer(numAct,'Name','fcAct2')
            tanhLayer('Name','tanhAct') ];

        actorOpts = rlRepresentationOptions('Optimizer','adam','LearnRate',1e-4);
        actor = rlContinuousDeterministicActorRepresentation(actorNetwork, obsInfo, actInfo, 'Observation',{'state'}, 'Action',{'tanhAct'}, actorOpts);

        % Critic network (Q-value)
        % Create separate input path for actions
        actionPath = [ featureInputLayer(numAct,'Normalization','none','Name','action') ];
        % Concatenate state and action
        criticNetwork = layerGraph(statePath);
        criticNetwork = addLayers(criticNetwork, actionPath);
        criticNetwork = addLayers(criticNetwork, concatenationLayer(1,2,'Name','concat'));
        criticNetwork = connectLayers(criticNetwork,'relu2','concat/in1');
        criticNetwork = connectLayers(criticNetwork,'action','concat/in2');
        % Add critic branches after concatenation
        criticLayers = [ fullyConnectedLayer(64,'Name','fcCritic1')
            reluLayer('Name','reluCritic1')
            fullyConnectedLayer(64,'Name','fcCritic2')
            reluLayer('Name','reluCritic2')
            fullyConnectedLayer(1,'Name','output') ];
        criticNetwork = addLayers(criticNetwork, criticLayers);
        criticNetwork = connectLayers(criticNetwork,'concat','fcCritic1');
        criticOpts = rlRepresentationOptions('Optimizer','adam','LearnRate',1e-3);
        critic = rlQValueRepresentation(criticNetwork, obsInfo, actInfo, 'Observation',{'state'}, 'Action',{'action'}, criticOpts);

    case {'PPO','A2C'}
        % Actor network (Gaussian policy)
        actorNetwork = layerGraph(statePath);
        % Mean output branch
        meanBranch = [ fullyConnectedLayer(32,'Name','fcMean1')
            reluLayer('Name','reluMean1')
            fullyConnectedLayer(numAct,'Name','mean') ];
        actorNetwork = addLayers(actorNetwork, meanBranch);
        actorNetwork = connectLayers(actorNetwork,'relu2','fcMean1');
        % Standard deviation output branch
        stdBranch = [ fullyConnectedLayer(32,'Name','fcStd1')
            reluLayer('Name','reluStd1')
            fullyConnectedLayer(numAct,'Name','std') ];
        actorNetwork = addLayers(actorNetwork, stdBranch);
        actorNetwork = connectLayers(actorNetwork,'relu2','fcStd1');
        % Representation for continuous Gaussian actor
        actorOpts = rlRepresentationOptions('Optimizer','adam','LearnRate',5e-4);
        actor = rlContinuousGaussianActorRepresentation(actorNetwork, obsInfo, actInfo, 'Observation',{'state'}, 'ActionMean',{'mean'}, 'ActionStd',{'std'}, actorOpts);

        % Critic network (state value function)
        criticNetwork = [ statePath
            fullyConnectedLayer(64,'Name','fcV1')
            reluLayer('Name','reluV1')
            fullyConnectedLayer(64,'Name','fcV2')
            reluLayer('Name','reluV2')
            fullyConnectedLayer(1,'Name','value') ];
        criticOpts = rlRepresentationOptions('Optimizer','adam','LearnRate',1e-3);
        critic = rlValueFunctionRepresentation(criticNetwork, obsInfo, 'Observation',{'state'}, criticOpts);

    otherwise
        error('Unsupported algorithm: %s', algName);
end

end

function [NextObservation, Reward, IsDone, LoggedSignals] = capStepWrapper(Action, LoggedSignals) %#ok<INUSL>
% capStepWrapper  Apply an action to the environment and return the next state and reward.
% The LoggedSignals input is unused but retained for compatibility with
% ``rlFunctionEnv``.  ``capStepFcn`` is expected to compute the
% environment dynamics and return the new observation, a scalar reward,
% a logical flag signalling episode termination, and any diagnostic
% information.
    [NextObservation, Reward, IsDone, LoggedSignals] = capStepFcn(Action);
end

%% Environment wrappers
function InitialObservation = capResetWrapper()
% capResetWrapper  Reset the underlying environment and return the initial observation.
    InitialObservation = capResetFcn();
end

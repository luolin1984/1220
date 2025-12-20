function [actor, critic1, critic2] = buildTD3Networks(obsInfo, actInfo, cap_min, cap_max)
% buildTD3Networks  Build actor + twin critics for TD3 with action scaling.
% Compatible with RL Toolbox versions where rlContinuousDeterministicActor only
% supports ObservationInputNames and UseDevice.

%% Dimensions
obsDim = prod(obsInfo.Dimension);
actDim = prod(actInfo.Dimension);

cap_min = cap_min(:);
cap_max = cap_max(:);

if numel(cap_min) == 1, cap_min = repmat(cap_min, actDim, 1); end
if numel(cap_max) == 1, cap_max = repmat(cap_max, actDim, 1); end

%% Actor network
actorLG = layerGraph();
actorLG = addLayers(actorLG, featureInputLayer(obsDim, "Normalization","none", "Name","obs"));
actorLG = addLayers(actorLG, fullyConnectedLayer(256, "Name","a_fc1"));
actorLG = addLayers(actorLG, reluLayer("Name","a_relu1"));
actorLG = addLayers(actorLG, fullyConnectedLayer(256, "Name","a_fc2"));
actorLG = addLayers(actorLG, reluLayer("Name","a_relu2"));
actorLG = addLayers(actorLG, fullyConnectedLayer(actDim, "Name","a_fc3"));
actorLG = addLayers(actorLG, tanhLayer("Name","a_tanh"));

scale = (cap_max - cap_min) / 2;
bias  = (cap_max + cap_min) / 2;
actorLG = addLayers(actorLG, scalingLayer("Name","a_scale", "Scale",scale, "Bias",bias));

actorLG = connectLayers(actorLG, "obs",     "a_fc1");
actorLG = connectLayers(actorLG, "a_fc1",   "a_relu1");
actorLG = connectLayers(actorLG, "a_relu1", "a_fc2");
actorLG = connectLayers(actorLG, "a_fc2",   "a_relu2");
actorLG = connectLayers(actorLG, "a_relu2", "a_fc3");
actorLG = connectLayers(actorLG, "a_fc3",   "a_tanh");
actorLG = connectLayers(actorLG, "a_tanh",  "a_scale");

actorNet = dlnetwork(actorLG);

try
    actor = rlContinuousDeterministicActor(actorNet, obsInfo, actInfo, ...
        ObservationInputNames="obs");
catch
    actorOpts = rlRepresentationOptions('LearnRate',1e-4,'GradientThreshold',1);
    actor = rlDeterministicActorRepresentation(actorLG, obsInfo, actInfo, ...
        'Observation', {'obs'}, 'Action', {'a_scale'}, actorOpts);
end

%% Critic template (Q(obs,act))
criticLG = layerGraph();
criticLG = addLayers(criticLG, featureInputLayer(obsDim, "Normalization","none", "Name","obs"));
criticLG = addLayers(criticLG, featureInputLayer(actDim, "Normalization","none", "Name","act"));
criticLG = addLayers(criticLG, concatenationLayer(1, 2, "Name","concat"));
criticLG = addLayers(criticLG, fullyConnectedLayer(256, "Name","c_fc1"));
criticLG = addLayers(criticLG, reluLayer("Name","c_relu1"));
criticLG = addLayers(criticLG, fullyConnectedLayer(256, "Name","c_fc2"));
criticLG = addLayers(criticLG, reluLayer("Name","c_relu2"));
criticLG = addLayers(criticLG, fullyConnectedLayer(1, "Name","q"));

criticLG = connectLayers(criticLG, "obs", "concat/in1");
criticLG = connectLayers(criticLG, "act", "concat/in2");
criticLG = connectLayers(criticLG, "concat", "c_fc1");
criticLG = connectLayers(criticLG, "c_fc1", "c_relu1");
criticLG = connectLayers(criticLG, "c_relu1", "c_fc2");
criticLG = connectLayers(criticLG, "c_fc2", "c_relu2");
criticLG = connectLayers(criticLG, "c_relu2", "q");

% Create TWO independent critics (different initial weights)
critic1 = makeCritic(criticLG, obsInfo, actInfo);
critic2 = makeCritic(criticLG, obsInfo, actInfo);

end

function critic = makeCritic(criticLG, obsInfo, actInfo)
criticNet = dlnetwork(criticLG);
try
    critic = rlQValueFunction(criticNet, obsInfo, actInfo, ...
        ObservationInputNames="obs", ActionInputNames="act");
catch
    criticOpts = rlRepresentationOptions('LearnRate',1e-3,'GradientThreshold',1);
    critic = rlQValueRepresentation(criticLG, obsInfo, actInfo, ...
        'Observation', {'obs'}, 'Action', {'act'}, criticOpts);
end
end

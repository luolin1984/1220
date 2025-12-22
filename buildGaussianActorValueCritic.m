function [actor, critic] = buildGaussianActorValueCritic(obsInfo, actInfo, cap_min, cap_max)
% 构造 Gaussian actor（mean/std）+ value critic
obsDim = prod(obsInfo.Dimension);
actDim = prod(actInfo.Dimension);

cap_min = cap_min(:); cap_max = cap_max(:);
scale = (cap_max - cap_min) / 2;
bias  = (cap_max + cap_min) / 2;

% ---- Actor (Gaussian) ----
lg = layerGraph();
lg = addLayers(lg, featureInputLayer(obsDim, "Normalization","none", "Name","obs"));
lg = addLayers(lg, fullyConnectedLayer(256, "Name","fc1"));
lg = addLayers(lg, reluLayer("Name","relu1"));
lg = addLayers(lg, fullyConnectedLayer(256, "Name","fc2"));
lg = addLayers(lg, reluLayer("Name","relu2"));

% mean head
lg = addLayers(lg, fullyConnectedLayer(actDim, "Name","mean_fc"));
lg = addLayers(lg, tanhLayer("Name","mean_tanh"));
lg = addLayers(lg, scalingLayer("Name","mean_scale", "Scale",scale, "Bias",bias));

% std head
lg = addLayers(lg, fullyConnectedLayer(actDim, "Name","std_fc"));
lg = addLayers(lg, softplusLayer("Name","std_sp"));
minStd = 0.05;
lg = addLayers(lg, scalingLayer("Name","std_bias", "Scale",ones(actDim,1), "Bias",minStd*ones(actDim,1)));

% connect trunk
lg = connectLayers(lg, "obs", "fc1");
lg = connectLayers(lg, "fc1", "relu1");
lg = connectLayers(lg, "relu1", "fc2");
lg = connectLayers(lg, "fc2", "relu2");

% branch to mean
lg = connectLayers(lg, "relu2", "mean_fc");
lg = connectLayers(lg, "mean_fc", "mean_tanh");
lg = connectLayers(lg, "mean_tanh", "mean_scale");

% branch to std
lg = connectLayers(lg, "relu2", "std_fc");
lg = connectLayers(lg, "std_fc", "std_sp");
lg = connectLayers(lg, "std_sp", "std_bias");

actorNet = dlnetwork(lg);

% 优先用新接口：rlContinuousGaussianActor
try
    actor = rlContinuousGaussianActor(actorNet, obsInfo, actInfo, ...
        ObservationInputNames="obs", ...
        ActionMeanOutputNames="mean_scale", ...
        ActionStandardDeviationOutputNames="std_bias");
catch
    % 兼容旧版 RL Toolbox（尽力而为）
    actorOpts = rlRepresentationOptions('LearnRate',1e-4,'GradientThreshold',1);
    actor = rlStochasticActorRepresentation(lg, obsInfo, actInfo, ...
        'Observation', {'obs'}, ...
        'Action', {'mean_scale','std_bias'}, actorOpts);
end

% ---- Critic (Value) ----
vLG = layerGraph();
vLG = addLayers(vLG, featureInputLayer(obsDim, "Normalization","none", "Name","obs"));
vLG = addLayers(vLG, fullyConnectedLayer(256, "Name","v_fc1"));
vLG = addLayers(vLG, reluLayer("Name","v_relu1"));
vLG = addLayers(vLG, fullyConnectedLayer(256, "Name","v_fc2"));
vLG = addLayers(vLG, reluLayer("Name","v_relu2"));
vLG = addLayers(vLG, fullyConnectedLayer(1, "Name","V"));

vLG = connectLayers(vLG, "obs", "v_fc1");
vLG = connectLayers(vLG, "v_fc1", "v_relu1");
vLG = connectLayers(vLG, "v_relu1", "v_fc2");
vLG = connectLayers(vLG, "v_fc2", "v_relu2");
vLG = connectLayers(vLG, "v_relu2", "V");

vNet = dlnetwork(vLG);

try
    critic = rlValueFunction(vNet, obsInfo, ObservationInputNames="obs");
catch
    criticOpts = rlRepresentationOptions('LearnRate',1e-3,'GradientThreshold',1);
    critic = rlValueRepresentation(vLG, obsInfo, ...
        'Observation', {'obs'}, criticOpts);
end

end
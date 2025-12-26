function cap_best = baseline_try_pso(env, lb, ub, x0, baseline, seed0)
cap_best = x0(:);
if exist('particleswarm','file') ~= 2
    warning('[baseline] particleswarm() not found -> skip PSO baseline.');
    return;
end

n = numel(lb);
try
    rng(seed0);
    obj = @(x) baseline_obj(x, env, baseline.MaxSteps, baseline.SearchEvalEpisodes, seed0+222);

    optsPSO = optimoptions('particleswarm', ...
        'Display','off', ...
        'SwarmSize', max(50, 10*n), ...
        'MaxIterations', 40, ...
        'FunctionTolerance', 1e-3);

    x = particleswarm(obj, n, lb', ub', optsPSO);
    cap_best = x(:);
catch ME
    warning('[baseline] PSO failed: %s', ME.message);
end
end
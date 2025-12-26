function cap_best = baseline_try_ga(env, lb, ub, x0, baseline, seed0)
cap_best = x0(:);
if exist('ga','file') ~= 2
    warning('[baseline] ga() not found -> skip GA baseline.');
    return;
end

n = numel(lb);
try
    rng(seed0);
    obj = @(x) baseline_obj(x, env, baseline.MaxSteps, baseline.SearchEvalEpisodes, seed0+111);

    optsGA = optimoptions('ga', ...
        'Display','off', ...
        'PopulationSize', max(40, 8*n), ...
        'MaxGenerations', 25, ...
        'FunctionTolerance', 1e-3);

    x = ga(obj, n, [],[],[],[], lb', ub', [], optsGA);
    cap_best = x(:);
catch ME
    warning('[baseline] GA failed: %s', ME.message);
end
end
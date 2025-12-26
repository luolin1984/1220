function cap_best = baseline_try_fmincon(env, lb, ub, x0, baseline, seed0)
cap_best = x0(:);
if exist('fmincon','file') ~= 2
    warning('[baseline] fmincon() not found -> skip fmincon baseline.');
    return;
end

try
    rng(seed0);
    obj = @(x) baseline_obj(x, env, baseline.MaxSteps, baseline.SearchEvalEpisodes, seed0+333);

    optsFC = optimoptions('fmincon', ...
        'Display','off', ...
        'Algorithm','sqp', ...
        'MaxIterations', 60, ...
        'OptimalityTolerance', 1e-3, ...
        'StepTolerance', 1e-6);

    x = fmincon(obj, x0(:), [],[],[],[], lb, ub, [], optsFC);
    cap_best = x(:);
catch ME
    warning('[baseline] fmincon failed: %s', ME.message);
end
end
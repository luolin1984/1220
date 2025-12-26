function [cap_best, hist] = baseline_random_search(env, lb, ub, N, nEvalEpisodes, maxSteps, seed0)
rng(seed0);
n = numel(lb);
bestJ = inf;
cap_best = (lb+ub)/2;

hist = struct();
hist.bestJ = nan(N,1);

for i = 1:N
    cap = lb + (ub-lb).*rand(n,1);
    J = baseline_obj(cap, env, maxSteps, nEvalEpisodes, seed0 + 10*i);
    if J < bestJ
        bestJ = J;
        cap_best = cap;
    end
    hist.bestJ(i) = bestJ;
end
end
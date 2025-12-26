function s = evaluateFixedCapPolicy(env, cap, nEpisodes, maxSteps, seed0)
% 用“固定 cap（开环）策略”跑 env，Return = sum(reward)
cap = cap(:);
rets = nan(nEpisodes,1);

for ep = 1:nEpisodes
    rng(seed0 + ep);         % 控制 reset 的随机性（如果你的 reset 用了 rng/randi）
    reset(env);

    G = 0;
    for t = 1:maxSteps
        try
            [~, r, isDone] = step(env, cap);
            if ~isfinite(r), r = -1e6; end
        catch
            r = -1e6;
            isDone = true;
        end

        G = G + r;
        if isDone, break; end
    end
    rets(ep) = G;
end

s.allReturns = rets;
s.meanReturn = mean(rets,'omitnan');
s.stdReturn  = std(rets,'omitnan');
s.bestReturn = max(rets);
end
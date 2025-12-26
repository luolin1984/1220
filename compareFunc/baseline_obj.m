function J = baseline_obj(capRow, env, maxSteps, nEvalEpisodes, seed0)
% 给优化器用的标量目标：最小化 J = -meanReturn
cap = capRow(:);
s = evaluateFixedCapPolicy(env, cap, nEvalEpisodes, maxSteps, seed0);
J = -s.meanReturn;
if ~isfinite(J), J = 1e12; end
end
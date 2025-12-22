function evalRes = evaluateAgentSimple(env, agent, nEpisodes, maxSteps)
% 手动评估：reset/step 循环，尽量避免版本差异
% 注：PPO/A2C 的策略天然带随机性，评价时是“抽样策略”的平均表现

% 尽量关闭探索（对 DDPG/TD3 有效）
try
    if isprop(agent, "UseExplorationPolicy")
        agent.UseExplorationPolicy = false;
    end
catch
end

returns = zeros(nEpisodes,1);
bestReturn = -inf;
bestInfo = struct();

for ep = 1:nEpisodes
    obs = reset(env);
    totalR = 0;

    logged = []; %#ok<NASGU>
    for t = 1:maxSteps
        action = getAction(agent, obs);
        [obs, r, done, loggedSignals] = step(env, action);
        totalR = totalR + double(r);

        if done
            break;
        end
    end

    returns(ep) = totalR;

    % 保存当前 episode 的最后一步信息（含 action_clip/kpis 等）
    if totalR > bestReturn
        bestReturn = totalR;
        if exist("loggedSignals","var") && isstruct(loggedSignals) && isfield(loggedSignals,"info")
            bestInfo = loggedSignals.info;
        else
            bestInfo = struct();
        end
    end
end

evalRes = struct();
evalRes.returns = returns;
evalRes.meanReturn = mean(returns);
evalRes.stdReturn  = std(returns);
evalRes.bestReturn = bestReturn;
evalRes.bestInfo   = bestInfo;

end
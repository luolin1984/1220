function opts = safeSetOpt(opts, field, value)
% 安全设置 options 字段，避免不同 MATLAB/RL 版本字段不一致直接报错
try
    opts.(field) = value;
catch
end
end
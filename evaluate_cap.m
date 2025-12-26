function [obs, reward, bad, loss, info] = evaluate_cap(action, prevAction, opts)
% evaluate_cap.m (v2: CVaR risk + CMDP constraint costs + Markov obs with dA)
% Outputs:
%   obs   : 5x1 normalized KPIs in [0,1]  [cost, curt, gas, vdev_CVaR, dA]
%   reward: base reward (without lambda); final reward computed in capStepFcn
%   bad   : hard failure flag
%   loss  : base loss (without lambda)
%   info  : diagnostics + constraint costs info.c

% defaults
obs    = ones(5,1);
reward = getfield_default(opts, 'bad_reward', -1.0);
bad    = 1;
loss   = 1.0;

info = struct();
info.stage = "init";

try
    info.stage = "read_bounds";
    cap_min = opts.cap_min;
    cap_max = opts.cap_max;

    % clip action
    info.stage = "clip_action";
    a  = local_to_col_double(action);
    pa = local_to_col_double(prevAction);

    n = numel(cap_min);

    a  = fit_to_n(a,  n, cap_min);
    pa = fit_to_n(pa, n, cap_min);

    a = min(max(a, cap_min), cap_max);
    info.action_clip = a;

    % action change penalty (normalized L1)
    info.stage = "compute_dA";
    denom = (cap_max - cap_min); denom(denom <= 0) = 1;
    dA = mean(abs(a - pa) ./ denom);
    info.dA = dA;

    % forward options into iter opts
    info.stage = "prepare_iter_opts";
    iter_opts = getfield_default(opts, 'iter_opts', struct());

    % optional scenario seed from reset (for multi-scenario training)
    if isfield(opts,'scenarioSeed')
        iter_opts.seed = opts.scenarioSeed;
    elseif isfield(iter_opts,'seed')
        % keep it
    else
        % allow deterministic if user wants
        iter_opts.seed = getfield_default(opts,'seed_base',1234);
    end
    info.iter_seed = iter_opts.seed;

    % adopt compressor options if present
    if isfield(opts, 'comp_ids'),         iter_opts.comp_ids = opts.comp_ids; end
    if isfield(opts, 'force_comp_as_el'), iter_opts.force_comp_as_el = opts.force_comp_as_el; end

    % call iter_couple
    info.stage = "before_iter";
    info.iter_path = string(which('iter_couple_most_mpng_24h_merged'));
    if strlength(info.iter_path) == 0
        error("iter_couple_most_mpng_24h_merged:not_on_path", ...
            "iter_couple_most_mpng_24h_merged.m 不在 MATLAB path 上。请 addpath(genpath(项目根目录)) 后重试。");
    end

    % make scenario reproducible (if iter_couple uses rng)
    try
        rng(double(iter_opts.seed));
    catch
    end

    w = getfield_default(opts, 'w', [1 1 1 1]);
    out = iter_couple_most_mpng_24h_merged(a, w, iter_opts);
    info.stage = "after_iter";

    % extract KPIs robustly
    info.stage = "extract_kpis";
    kpis = struct();
    if isstruct(out) && isfield(out, 'kpis') && isstruct(out.kpis)
        kpis = out.kpis;
    end
    info.kpis = kpis;

    % detect hard failure
    info.stage = "detect_fail";
    hard_fail = false;
    if isstruct(out) && isfield(out,'most_ok')
        hard_fail = hard_fail || (~logical(out.most_ok));
    end
    if isstruct(out) && isfield(out,'gas_ok')
        hard_fail = hard_fail || (~logical(out.gas_ok));
    end

    % KPI numbers
    info.stage = "read_kpi_nums";
    avg_cost = get_kpi_num(kpis, 'avg_cost', NaN);
    curt     = get_kpi_num(kpis, 'curtail_ratio', NaN);
    gas_risk = get_kpi_num(kpis, 'gas_risk', NaN);

    % --- Voltage risk series (preferred) ---
    info.stage = "read_vdev_series";
    vdev_series = get_kpi_series(kpis, {'voltage_dev_series_pu','voltage_dev_pu','vdev_series','vdev_raw_series'});
    % fallback to scalar
    vdev_raw = get_kpi_num(kpis, 'voltage_dev_avg_pu', NaN);
    if isempty(vdev_series)
        if isfinite(vdev_raw)
            vdev_series = vdev_raw; % scalar fallback
            info.vdev_src = "scalar:out.kpis.voltage_dev_avg_pu";
        else
            vdev_series = NaN;
            info.vdev_src = "vdev:missing";
        end
    else
        info.vdev_src = "series:kpis.(voltage_dev_*)";
    end

    % CVaR (top-tail mean)
    norm = getfield_default(opts,'norm',struct());
    alpha = getfield_default(norm,'cvar_alpha', 0.1); % top 10% tail
    vdev_cvar = cvar_top_mean(vdev_series, alpha);
    info.vdev_cvar = vdev_cvar;

    % clip & normalize (note: vdev_cap should be a physical cap e.g. 0.2 pu)
    vdev_cap  = getfield_default(norm, 'vdev_cap', 0.2);
    vdev_clip = min(max(vdev_cvar, 0), vdev_cap);
    info.vdev_clip = vdev_clip;

    % normalize obs to [0,1]
    info.stage = "normalize_obs";
    cost_ref = getfield_default(norm,'cost_ref', 1);
    curt_ref = getfield_default(norm,'curt_ref', 1e-3);
    gas_ref  = getfield_default(norm,'gas_ref',  1e-3);

    cost_n = clip01(avg_cost / max(cost_ref, eps));
    curt_n = clip01(curt     / max(curt_ref, eps));
    gas_n  = clip01(gas_risk / max(gas_ref,  eps));
    vdev_n = clip01(vdev_clip / max(vdev_cap, eps));

    % Markovization: include dA in obs
    dA_cap = getfield_default(norm,'dA_cap', 1.0); % since dA is already normalized mean L1, cap in [0,1]
    dA_n   = clip01(dA / max(dA_cap, eps));

    obs = [cost_n; curt_n; gas_n; vdev_n; dA_n];

    % base loss and base reward (WITHOUT lambda; lambda handled in capStepFcn)
    info.stage = "loss_reward_base";
    loss = mean(obs(1:4)); % do not include dA directly here; it's part of state, not objective
    base_reward = max(min(1 - loss, 1), -1);

    reward = base_reward;

    % --- CMDP constraint costs (non-negative) ---
    % constraints: vdev_CVaR <= vdev_limit, gas_risk <= gas_limit
    info.stage = "cmdp_costs";
    constr = getfield_default(opts, 'constr', struct());

    vdev_limit = getfield_default(constr, 'vdev_limit', 0.05); % pu, example
    gas_limit  = getfield_default(constr, 'gas_limit',  0.10); % example (normalize to your gas_risk meaning)

    c1 = max(0, (vdev_cvar - vdev_limit) / max(vdev_limit, eps));
    c2 = max(0, (gas_risk  - gas_limit ) / max(gas_limit,  eps));

    info.c = [c1; c2];
    info.vdev_limit = vdev_limit;
    info.gas_limit  = gas_limit;

    % finalize bad flag
    bad = hard_fail || any(~isfinite(obs)) || any(~isfinite(info.c));
    if bad
        reward = getfield_default(opts,'bad_reward', -1.0);
    end

    info.stage = "done_ok";

catch ME
    if getfield_default(opts,'debug_rethrow',false)
        rethrow(ME);
    end
    bad    = 1;
    obs    = ones(5,1);
    loss   = 1.0;
    reward = getfield_default(opts,'bad_reward', -1.0);

    info.err_id  = string(ME.identifier);
    info.err_msg = string(ME.message);
    info.err     = getReport(ME, "extended", "hyperlinks", "off");

    if ~isfield(info,'iter_path')
        info.iter_path = string(which('iter_couple_most_mpng_24h_merged'));
    end
    if ~isfield(info,'stage')
        info.stage = "unknown";
    end
    info.c = [1;1];
end
end

%% ---- helpers ----
function x = get_kpi_num(kpis, name, d)
x = d;
if isstruct(kpis) && isfield(kpis, name)
    v = kpis.(name);
    if isnumeric(v) && isscalar(v) && isfinite(v)
        x = double(v);
    elseif isnumeric(v) && ~isempty(v) && all(isfinite(v(:)))
        x = double(mean(v(:)));
    end
end
end

function s = get_kpi_series(kpis, names)
% return numeric vector if any candidate field exists
s = [];
if ~isstruct(kpis), return; end
for i = 1:numel(names)
    nm = names{i};
    if isfield(kpis, nm)
        v = kpis.(nm);
        if isnumeric(v) && ~isempty(v)
            vv = double(v(:));
            vv = vv(isfinite(vv));
            if ~isempty(vv)
                s = vv;
                return;
            end
        end
    end
end
end

function y = clip01(x)
if ~isfinite(x), y = 1; return; end
y = min(max(double(x), 0), 1);
end

function v = getfield_default(s, f, d)
if isstruct(s) && isfield(s, f)
    v = s.(f);
else
    v = d;
end
end

function v = local_to_col_double(x)
while iscell(x) && numel(x)==1, x = x{1}; end
if iscell(x)
    try
        x = cell2mat(x(:));
    catch
        x = cellfun(@(z) double(z), x(:));
    end
end
if isa(x,'dlarray'),  x = extractdata(x); end
if isa(x,'gpuArray'), x = gather(x);      end
v = double(x(:));
end

function v = fit_to_n(v, n, fill_with)
% ensure column vector length n
v = v(:);
if numel(v) == n, return; end
if isscalar(v)
    v = repmat(v, n, 1);
    return;
end
v = v(1:min(end,n));
if numel(v) < n
    v(end+1:n,1) = fill_with(numel(v)+1:n);
end
end

function m = cvar_top_mean(x, alpha)
% CVaR: mean of top alpha fraction (tail risk)
if isempty(x) || all(~isfinite(x))
    m = NaN; return;
end
xx = double(x(:));
xx = xx(isfinite(xx));
if isempty(xx), m = NaN; return; end
xx = sort(xx, 'descend');
k = max(1, ceil(alpha * numel(xx)));
m = mean(xx(1:k));
end

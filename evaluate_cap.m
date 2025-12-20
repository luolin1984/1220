function [obs, reward, bad, loss, info] = evaluate_cap(action, prevAction, opts)
% evaluate_cap_vdev4.m
% Returns:
%   obs   : 4x1 normalized KPIs in [0,1]
%   reward: shaped reward in [-1, 1]
%   bad   : hard-failure flag (1 means "treat as invalid transition")
%   loss  : weighted loss in [0, +inf) (smaller is better)
%   info  : diagnostics (vdev_raw/vdev_clip/src, dA, kpis, etc.)

% defaults (safe)
obs    = ones(4,1);
reward = getfield_default(opts, 'bad_reward', -1.0);
bad    = 1;
loss   = 1.0;

info = struct();
info.action_in   = double(action(:));
info.action_clip = double(action(:));
info.dA          = NaN;
info.vdev_raw    = NaN;
info.vdev_clip   = getfield_default(getfield_default(opts,'norm',struct()), 'vdev_cap', 0.2);
info.vdev_src    = "evaluate_cap:init";
info.kpis        = struct();
info.err         = "";

try
    cap_min = opts.cap_min;
    cap_max = opts.cap_max;

    % clip action
    a = double(action(:));
    a = min(max(a, cap_min), cap_max);
    info.action_clip = a;

    % action change penalty (normalized L1)
    denom = (cap_max - cap_min);
    denom(denom <= 0) = 1;
    info.dA = mean(abs(a - double(prevAction(:))) ./ denom);

    % forward compressor options into iter opts
    iter_opts = getfield_default(opts, 'iter_opts', struct());
    if isfield(opts, 'comp_ids'),         iter_opts.comp_ids = opts.comp_ids; end
    if isfield(opts, 'force_comp_as_el'), iter_opts.force_comp_as_el = opts.force_comp_as_el; end

    % call iter_couple
    w = getfield_default(opts, 'w', [1 1 1 1]);
    out = iter_couple_most_mpng_24h_merged(a, w, iter_opts);

    % extract KPIs robustly
    kpis = struct();
    if isstruct(out) && isfield(out, 'kpis') && isstruct(out.kpis)
        kpis = out.kpis;
    end
    info.kpis = kpis;

    % detect hard failure from out flags (if present)
    hard_fail = false;
    if isstruct(out) && isfield(out,'most_ok')
        hard_fail = hard_fail || (~logical(out.most_ok));
    end
    if isstruct(out) && isfield(out,'gas_ok')
        hard_fail = hard_fail || (~logical(out.gas_ok));
    end

    % pull KPI numbers with safe fallbacks (soft-fail, not hard-fail)
    avg_cost = get_kpi_num(kpis, 'avg_cost', NaN);
    curt     = get_kpi_num(kpis, 'curtail_ratio', NaN);
    gas_risk = get_kpi_num(kpis, 'gas_risk', NaN);

    % voltage deviation (preferred: kpis.voltage_dev_avg_pu)
    vdev_raw = get_kpi_num(kpis, 'voltage_dev_avg_pu', NaN);
    if isfinite(vdev_raw)
        info.vdev_src = "out.kpis.voltage_dev_avg_pu";
    else
        % optional alternative names (if your iter function uses them)
        vdev_raw = get_kpi_num(kpis, 'vdev_raw', NaN);
        if isfinite(vdev_raw)
            info.vdev_src = "out.kpis.vdev_raw";
        else
            info.vdev_src = "vdev:missing";
        end
    end

    vdev_cap  = getfield_default(getfield_default(opts,'norm',struct()), 'vdev_cap', 0.2);
    vdev_clip = min(max(vdev_raw, 0), vdev_cap);
    if ~isfinite(vdev_raw)
        vdev_raw  = vdev_cap;      % worst-but-finite fallback
        vdev_clip = vdev_cap;
    end
    info.vdev_raw  = vdev_raw;
    info.vdev_clip = vdev_clip;

    % if any KPI missing, treat as soft-bad (penalize), but do not hard-fail
    soft_bad = (~isfinite(avg_cost)) || (~isfinite(curt)) || (~isfinite(gas_risk));
    if soft_bad
        % fill missing with worst-but-finite values
        if ~isfinite(avg_cost), avg_cost = getfield_default(getfield_default(opts,'norm',struct()), 'cost_ref', 30); end
        if ~isfinite(curt),     curt     = 1.0; end
        if ~isfinite(gas_risk), gas_risk = getfield_default(getfield_default(opts,'norm',struct()), 'gas_ref', 1e-3); end
    end

    % optionally upgrade invalid KPI to hard fail
    if getfield_default(opts,'hard_fail_on_invalid_kpi', false) && soft_bad
        hard_fail = true;
    end

    % normalize into obs in [0,1]
    norm = getfield_default(opts,'norm',struct());
    cost_ref = getfield_default(norm,'cost_ref', 30);
    curt_ref = getfield_default(norm,'curt_ref', 1.0);
    gas_ref  = getfield_default(norm,'gas_ref', 1e-3);

    cost_n = clip01(avg_cost / max(cost_ref, eps));
    curt_n = clip01(curt     / max(curt_ref, eps));
    gas_n  = clip01(gas_risk / max(gas_ref,  eps));
    vdev_n = clip01(vdev_clip / max(vdev_cap, eps));

    obs = [cost_n; curt_n; gas_n; vdev_n];

    % loss and reward
    % (simple: higher reward when KPIs small)
    loss = mean(obs);

    dA_lambda = getfield_default(norm,'dA_lambda', 0.0);
    loss = loss + dA_lambda * info.dA;

    reward = 1.0 - loss;              % roughly in [-inf, 1]
    reward = max(min(reward, 1.0), -1.0);

    % hard fail override
    if hard_fail
        bad    = 1;
        obs    = ones(4,1);
        loss   = 1.0;
        reward = getfield_default(opts,'bad_reward', -1.0);
        info.vdev_src = string(info.vdev_src) + ":hard_fail";
    else
        bad = 0;
    end

catch ME
    bad    = 1;
    obs    = ones(4,1);
    loss   = 1.0;
    reward = getfield_default(opts,'bad_reward', -1.0);

    info.err = getReport(ME, "basic", "hyperlinks", "off");
    info.vdev_src = "evaluate_cap:exception";
    info.vdev_raw = NaN;
    info.vdev_clip = getfield_default(getfield_default(opts,'norm',struct()), 'vdev_cap', 0.2);
end
end

function x = get_kpi_num(kpis, name, d)
x = d;
if isstruct(kpis) && isfield(kpis, name)
    v = kpis.(name);
    if isnumeric(v) && isscalar(v)
        x = double(v);
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

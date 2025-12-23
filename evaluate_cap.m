function [obs, reward, bad, loss, info] = evaluate_cap(action, prevAction, opts)
% evaluate_cap.m (patched: debug+stage, safer exception reporting)
% Returns:
%   obs   : 4x1 normalized KPIs in [0,1]
%   reward: shaped reward in [-1, 1]
%   bad   : hard-failure flag (1 means "treat as invalid transition")
%   loss  : weighted loss in [0, +inf)
%   info  : diagnostics (vdev_raw/vdev_clip/src, dA, stage, err, etc.)

% defaults (safe)
obs    = ones(4,1);
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
    
    % a = double(action(:));

    % --- robust action parsing (cell/dlarray/gpuArray -> double column) ---
    a  = action;
    pa = prevAction;

    a  = local_to_double_col(a);
    pa = local_to_double_col(pa);

    a = min(max(a, cap_min), cap_max);
    info.action_clip = a;

    % action change penalty (normalized L1)
    info.stage = "compute_dA";
    denom = (cap_max - cap_min);
    denom(denom <= 0) = 1;
    info.dA = mean(abs(a - pa) ./ denom);

    % forward compressor options into iter opts
    info.stage = "prepare_iter_opts";
    iter_opts = getfield_default(opts, 'iter_opts', struct());
    if isfield(opts, 'comp_ids'),         iter_opts.comp_ids = opts.comp_ids; end
    if isfield(opts, 'force_comp_as_el'), iter_opts.force_comp_as_el = opts.force_comp_as_el; end

    % call iter_couple
    info.stage = "before_iter";
    info.iter_path = string(which('iter_couple_most_mpng_24h_merged'));
    if strlength(info.iter_path) == 0
        error("iter_couple_most_mpng_24h_merged:not_on_path", ...
            "iter_couple_most_mpng_24h_merged.m 不在 MATLAB path 上。请 addpath(genpath(项目根目录)) 后重试。");
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

    % detect hard failure from out flags (if present)
    info.stage = "detect_fail";
    hard_fail = false;
    if isstruct(out) && isfield(out,'most_ok')
        hard_fail = hard_fail || (~logical(out.most_ok));
    end
    if isstruct(out) && isfield(out,'gas_ok')
        hard_fail = hard_fail || (~logical(out.gas_ok));
    end

    % pull KPI numbers with safe fallbacks
    info.stage = "read_kpi_nums";
    avg_cost = get_kpi_num(kpis, 'avg_cost', NaN);
    curt     = get_kpi_num(kpis, 'curtail_ratio', NaN);
    gas_risk = get_kpi_num(kpis, 'gas_risk', NaN);

    % voltage deviation (preferred: kpis.voltage_dev_avg_pu)
    info.stage = "read_vdev";
    vdev_raw = get_kpi_num(kpis, 'voltage_dev_avg_pu', NaN);
    if isfinite(vdev_raw)
        info.vdev_src = "out.kpis.voltage_dev_avg_pu";
    else
        vdev_raw = get_kpi_num(kpis, 'vdev_raw', NaN);
        if isfinite(vdev_raw)
            info.vdev_src = "out.kpis.vdev_raw";
        else
            info.vdev_src = "vdev:missing";
        end
    end

    norm = getfield_default(opts,'norm',struct());
    vdev_cap  = getfield_default(norm, 'vdev_cap', 0.2);
    vdev_clip = min(max(vdev_raw, 0), vdev_cap);
    info.vdev_raw  = vdev_raw;
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

    obs = [cost_n; curt_n; gas_n; vdev_n];

    % loss and reward
    info.stage = "loss_reward";
    loss = mean(obs);

    dA_lambda = getfield_default(norm,'dA_lambda', 0.0);
    if isfinite(info.dA)
        loss = loss + dA_lambda * info.dA;
    end

    % reward in [-1,1] (simple: negative loss; clamp)
    reward = max(min(1 - loss, 1), -1);

    % finalize bad flag
    bad = hard_fail || any(~isfinite(obs));
    if bad
        reward = getfield_default(opts,'bad_reward', -1.0);
    end

    info.stage = "done_ok";

catch ME
    if getfield_default(opts,'debug_rethrow',false)
        rethrow(ME);
    end
    bad    = 1;
    obs    = ones(4,1);
    loss   = 1.0;
    reward = getfield_default(opts,'bad_reward', -1.0);

    info.err_id  = string(ME.identifier);
    info.err_msg = string(ME.message);
    info.err     = getReport(ME, "extended", "hyperlinks", "off");

    if ~isfield(info,'vdev_src') || isempty(info.vdev_src)
        info.vdev_src = "evaluate_cap:exception";
    else
        info.vdev_src = string(info.vdev_src) + ":evaluate_cap:exception";
    end
    info.vdev_raw  = NaN;
    info.vdev_clip = getfield_default(getfield_default(opts,'norm',struct()), 'vdev_cap', 0.2);

    if ~isfield(info,'iter_path')
        info.iter_path = string(which('iter_couple_most_mpng_24h_merged'));
    end
    if ~isfield(info,'stage')
        info.stage = "unknown";
    end
end
end

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

function x = local_to_double_col(x)
    if isa(x, 'dlarray')
        x = extractdata(x);
    end
    if isa(x, 'gpuArray')
        x = gather(x);
    end
    if iscell(x)
        if numel(x) == 1
            x = x{1};
        else
            % 既兼容 {scalar,scalar,...} 也兼容 {vector}
            x = cell2mat(x(:));
        end
    end
    x = double(x(:));
end

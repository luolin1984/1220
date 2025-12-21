function out = iter_couple_most_mpng_24h_merged(cap_in, w_obj, opts)
% 功能：在 24 小时滚动时域内，对“电网 + 天然气网”进行 MOST+MPNG 迭代耦合调度。
%      - 电侧：采用 MOST（DC OPF），支持风电、双光伏、燃气机组、平衡机组和可选储能
%      - 气侧：采用 MPNG，对 48 节点气网做稳态潮流和压缩机工况计算
%      - 耦合：将电/气互联系统的关系（燃气机 → 气井/管网，电驱压缩机 → 电网负荷）
%              通过 connect 结构体描述，并在 6.5) 步骤实现“强耦合迭代”
%
% 输入：
%   cap_in : 各电源/设备容量向量（一般作为 DRL/优化的决策变量），内部按固定顺序解析
%   w_obj  : 多目标权重向量 [w_econ, w_curtail, w_smooth]
%   opts   : 结构体，可选参数，例如：
%            - opts.comp_bus        : 压缩机等效到哪个电网母线 ('wind' / 'pv1' / bus号)
%            - opts.comp_ids        : 参与耦合的压缩机编号（向量或 'all'）
%            - opts.comp_pc_unit    : mgc.comp(:,PC_C) 的口径 ('MW' | 'MWh/day' | 'auto')
%            - opts.storage         : 储能配置（见下方 1) 储能挂接）
%
% 输出：
%   out.obj          : [obj_econ, obj_curtail, obj_smooth] 多目标指标（越大越好）
%   out.score        : 按 w_obj 聚合后的单一评分
%   out.cost_total   : 24h 总运行成本（$）
%   out.curtail_MWh  : 24h 可再生能量弃风/弃光电量（MWh）
%   out.comp_el_MW   : 压缩机各台在各时段的“等效电功率”矩阵 (n_comp × T)
%   out.kpis         : 其他便于分析的指标合集（利用率、成本口径等）
%   out.most_ok      : MOST 是否收敛的布尔标志
%   out.Pg_storage_* : 如存在储能，返回其 24h 的功率曲线（正=放电，负=充电）

%%% ------------------- 参数 -------------------
if nargin < 3 || isempty(opts), opts = struct(); end

% ==== 压缩机耦合相关 ====
% comp_bus     : 压缩机等效为“附加负荷”时挂在哪类母线
% comp_ids     : 参与耦合的压缩机编号 （'all' 或 [1 3 5]）
% force_comp_as_el : 即使不是 COMP_P 类型，也强制按“电驱压缩机”处理
if ~isfield(opts,'comp_bus')      || isempty(opts.comp_bus),      opts.comp_bus      = 'wind'; end
if ~isfield(opts,'comp_ids')      || isempty(opts.comp_ids),      opts.comp_ids      = 'all';  end   % 'all' 或如 [1 3 5]
if ~isfield(opts,'force_comp_as_el') || isempty(opts.force_comp_as_el), opts.force_comp_as_el = false; end
if ~isfield(opts,'comp_pc_unit')  || isempty(opts.comp_pc_unit),  opts.comp_pc_unit  = 'auto'; end % 'auto'|'MW'|'MWh/day'
if ~isfield(opts,'do_plot') || isempty(opts.do_plot)
    opts.do_plot = false;   % 默认不画图
end

% ==== 储能配置（可选） ====
% opts.storage.wind / opts.storage.pv1 均为结构体，可包含字段：
%   - bus  : 挂接母线号（缺省用风电/PV1 的母线）
%   - Pmax : 充/放电功率上限（MW），函数内部会自动设置 PMIN = -Pmax
%   - Emax : 能量上限（MWh），缺省为“4 小时额定功率”：Emax = 4 * Pmax
%   - SOC0 : 初始 SOC（0~1），缺省 0.5
if ~isfield(opts,'storage') || isempty(opts.storage)
    opts.storage = struct();    % 默认为无储能
end

T = 24;

assert(exist('loadmd','file')==2, '未检测到 MOST (loadmd)。');
assert(exist('loadstoragedata','file')==2, '未检测到 MOST 储能模块 (loadstoragedata)。');
assert(exist('mpoption','file')==2 && exist('runopf','file')==2, '未检测到 MATPOWER。');

% MOST 配置（DC，无UC）
mpopt_most = mpoption('most.dc_model', 1, 'most.uc.run', 0, 'verbose', 2);
try mpopt_most = mpoption(mpopt_most, 'most.enforce_reserves', 0); catch, end
try mpopt_most = mpoption(mpopt_most, 'most.storage.cyclic', 0);   catch, end
try mpopt_most = mpoption(mpopt_most, 'most.enforce_ramps', 1); catch, end

% ====== MPNG 调用配置：保留但优先使用“单参数”签名 ======
mpopt_mpng_AC = mpoption('opf.ac.solver', 'MIPS', 'verbose', 1);
mpopt_mpng_AC = mpoption(mpopt_mpng_AC, ...
    'pf.enforce_q_lims', 0, ...
    'opf.flow_lim', 'S', ...
    'opf.start', 'DC', ...
    'mips.step_control', 1);

mpopt_mpng_IPOPT = mpoption(mpopt_mpng_AC, 'opf.ac.solver', 'IPOPT');
try
    mpopt_mpng_IPOPT = mpoption(mpopt_mpng_IPOPT, ...
        'ipopt.opts.max_iter', 4000, ...
        'ipopt.opts.tol', 1e-7);
catch
end

mpopt_mpng_DC = mpoption('model','DC','opf.dc.solver','MIPS','verbose',1);

% 索引
define_constants;
[CT_LABEL, CT_PROB, CT_TABLE, ...
    CT_TBUS, CT_TGEN, CT_TBRCH, CT_TAREABUS, CT_TAREAGEN, CT_TAREABRCH, ...
    CT_ROW, CT_COL, CT_CHGTYPE, CT_REP, CT_REL, CT_ADD, CT_NEWVAL, ...
    CT_TLOAD, CT_TGENCOST, CT_TAREALOAD, CT_MODCOST, ...
    CT_LOAD_ALL_P, CT_LOAD_FIX_P] = idx_ct; %#ok<ASGLU>
ID = get_indices();

%% ------------------- 1) 电网底盘 + 机组 -------------------
mpc = loadcase('case33bw');
NB  = size(mpc.bus,1);
% 放宽支路限额
if size(mpc.branch,2) >= ID.RATE_A
    mpc.branch(:, ID.RATE_A) = max(mpc.branch(:, ID.RATE_A), 1e4);
end

% 区域表
if ~isfield(mpc, 'areas') || isempty(mpc.areas), mpc.areas = [1 1]; end
if size(mpc.bus,2) >= ID.BUS_AREA, mpc.bus(:, ID.BUS_AREA) = 1; end

% 1) 解析容量
Pwind   = cap_in(1);
Ppv1    = cap_in(2);
Ppv2    = cap_in(3);
Pgt_max = cap_in(5);

% 2) 默认母线（兼容老版本：没有选址动作时，仍然用原来的 8/13/30）
bus_wind = 8;
bus_pv1  = 13;
bus_pv2  = 30;
bus_gas  = 30;     % 燃机继续挂在 30 节点

% 3) 若 cap_in 长度 ≥ 7/8/9，则认为 7/8/9 分别为风/光1/光2 的“候选母线号”
if numel(cap_in) >= 7
    bw_raw = cap_in(7);
    bus_wind = max(1, min(NB, round(bw_raw)));
end
if numel(cap_in) >= 8
    bp1_raw = cap_in(8);
    bus_pv1 = max(1, min(NB, round(bp1_raw)));
end
if numel(cap_in) >= 9
    bp2_raw = cap_in(9);
    bus_pv2 = max(1, min(NB, round(bp2_raw)));
end

fprintf('[SITE] WIND@bus=%d, PV1@bus=%d, PV2@bus=%d, GAS@bus=%d\n', ...
    bus_wind, bus_pv1, bus_pv2, bus_gas);

[mpc, idx_wind] = safe_add_gen(mpc, bus_wind, Pwind,   0.6,  1.00, 'wind',    ID);
[mpc, idx_pv1 ] = safe_add_gen(mpc, bus_pv1,  Ppv1,    0.0,  1.00, 'pv',      ID);
[mpc, idx_pv2 ] = safe_add_gen(mpc, bus_pv2,  Ppv2,    0.0,  1.00, 'pv',      ID);
[mpc, idx_gas ] = safe_add_gen(mpc, bus_gas,  Pgt_max, 2.0,  1.00, 'thermal', ID);
mpc.gencost(idx_gas, :) = pad_gencost_linear(mpc.gencost, 45);

%% ===== [PATCH B1+B2] 现实化上限 & 兜底收窄 =====
Pgt_max = max(Pgt_max, 40.0);
mpc.gen(idx_gas, ID.PMAX) = Pgt_max;
mpc.gencost(idx_gas, :)   = pad_gencost_linear(mpc.gencost, 45);

% 高价兜底机
[mpc, idx_slack] = safe_add_gen(mpc, 1, 500.0, 10.0, 1.00, 'thermal', ID);
mpc.gencost(idx_slack, :) = pad_gencost_linear(mpc.gencost, 2000);
mpc.gen(idx_slack, ID.PMAX) = 1.0;
mpc.gencost(idx_slack, :)   = pad_gencost_linear(mpc.gencost, 2000);

% ===== 新增：主网机组（case33bw 原有的第 1 行）视为 GRID，调整成本和容量 =====
idx_grid = 1;                         % case33bw 原始机组行
Pd_base  = sum(mpc.bus(:, ID.PD));    % 当前基准负荷 (MW)

if idx_grid >= 1 && idx_grid <= size(mpc.gen,1)
    % 至少能单独承担 120% 的负荷，避免因为 PMAX 太小导致兜底机频繁出力
    mpc.gen(idx_grid, ID.PMAX) = max(mpc.gen(idx_grid, ID.PMAX), 1.2 * Pd_base);

    % 成本顺序：风/光≈0 < 燃气(45) < GRID(60) < 兜底机(2000)
    % 若你希望“购电比自备机便宜”，可以把 60 改成 40。
    mpc.gencost(idx_grid, :) = pad_gencost_linear(mpc.gencost, 60);
end

% ===== 在风电、PV1 母线挂接储能（MOST-SOC 版本） =====
% 思路：在现有风电/PV1 母线旁边“挂”一个虚拟机组，功率可以正负（放电/充电），
%      然后在 MOST 的 sd（storage data）里为这些虚拟机组建立能量约束（SOC 约束）。
idx_ebat_wind = [];
idx_ebat_pv1  = [];

% 用三个数组把“哪些 gen 行是储能”以及它们的能量参数记下来
ess_rows = [];     % 对应 gen 行号（UnitIdx）
ess_Emin = [];     % 最小能量 (MWh)
ess_Emax = [];     % 最大能量 (MWh)
ess_E0   = [];     % 初始能量 (MWh)

% 全局效率：既可以直接给 eta_rt，也可以分别给 eta_ch / eta_dis
eta_ch  = 1.0;     % 充电效率
eta_dis = 1.0;     % 放电效率
if isfield(opts,'storage') && ~isempty(opts.storage)
    if isfield(opts.storage,'eta_rt') && ~isempty(opts.storage.eta_rt)
        eta_rt  = opts.storage.eta_rt;
        eta_ch  = sqrt(eta_rt);
        eta_dis = sqrt(eta_rt);
    end
    if isfield(opts.storage,'eta_ch') && ~isempty(opts.storage.eta_ch)
        eta_ch = opts.storage.eta_ch;
    end
    if isfield(opts.storage,'eta_dis') && ~isempty(opts.storage.eta_dis)
        eta_dis = opts.storage.eta_dis;
    end
end

if ~isempty(opts.storage)
    % --- 风电侧储能 ---
    if isfield(opts.storage,'wind') && ~isempty(opts.storage.wind)
        st = opts.storage.wind;
        if ~isfield(st,'Pmax') || isempty(st.Pmax), st.Pmax = 0; end
        if ~isfield(st,'bus')  || isempty(st.bus),  st.bus  = bus_wind; end
        % Emax（能量上限）缺省设为“4 小时额定功率”
        if ~isfield(st,'Emax') || isempty(st.Emax)
            st.Emax = 4 * st.Pmax;
        end
        % 初始 SOC（0~1），默认 0.5
        if ~isfield(st,'SOC0') || isempty(st.SOC0)
            st.SOC0 = 0.5;
        end

        if st.Pmax > 0
            [mpc, idx_ebat_wind] = safe_add_gen(mpc, st.bus, st.Pmax, 0.0, 1.00, 'storage', ID);
            % 允许充电：负出力视作负荷
            mpc.gen(idx_ebat_wind, ID.PMIN) = -st.Pmax;
            % 成本很小，鼓励其在有可再生富余时吸收/释放
            mpc.gencost(idx_ebat_wind, :)   = pad_gencost_linear(mpc.gencost, 1);

            % 收集能量参数（MWh）
            Emin = 0;
            Emax = st.Emax;
            E0   = max(Emin, min(Emax, st.SOC0 * Emax));

            ess_rows(end+1,1) = idx_ebat_wind;
            ess_Emin(end+1,1) = Emin;
            ess_Emax(end+1,1) = Emax;
            ess_E0(end+1,1)   = E0;
        end
    end

    % --- PV1 侧储能 ---
    if isfield(opts.storage,'pv1') && ~isempty(opts.storage.pv1)
        st = opts.storage.pv1;
        if ~isfield(st,'Pmax') || isempty(st.Pmax), st.Pmax = 0; end
        if ~isfield(st,'bus')  || isempty(st.bus),  st.bus  = bus_pv1; end
        if ~isfield(st,'Emax') || isempty(st.Emax)
            st.Emax = 4 * st.Pmax;
        end
        if ~isfield(st,'SOC0') || isempty(st.SOC0)
            st.SOC0 = 0.5;
        end

        if st.Pmax > 0
            [mpc, idx_ebat_pv1] = safe_add_gen(mpc, st.bus, st.Pmax, 0.0, 1.00, 'storage', ID);
            mpc.gen(idx_ebat_pv1, ID.PMIN) = -st.Pmax;
            mpc.gencost(idx_ebat_pv1, :)   = pad_gencost_linear(mpc.gencost, 1);

            Emin = 0;
            Emax = st.Emax;
            E0   = max(Emin, min(Emax, st.SOC0 * Emax));

            ess_rows(end+1,1) = idx_ebat_pv1;
            ess_Emin(end+1,1) = Emin;
            ess_Emax(end+1,1) = Emax;
            ess_E0(end+1,1)   = E0;
        end
    end
end

% 补齐 gencost 行数（原逻辑保留）
if size(mpc.gencost,1) < size(mpc.gen,1)
    need = size(mpc.gen,1) - size(mpc.gencost,1);
    mpc.gencost = [mpc.gencost; repmat(mpc.gencost(1,:), need, 1)];
end

% ===== 基于 StorageDataTable 构造 MOST 储能数据 sd =====
sd  = [];
xgd = [];   % 目前不使用 xGenData 的高级特性，先留空

if ~isempty(ess_rows)
    storage_sd_table = struct();

    % 统一的充放电效率（所有储能单元共用）
    storage_sd_table.OutEff     = eta_dis;
    storage_sd_table.InEff      = eta_ch;
    storage_sd_table.LossFactor = 0;    % 不考虑自放电
    storage_sd_table.rho        = 0;    % 纯期望值模型

    % 每个储能单元的能量参数（行对应一个 UnitIdx）
    storage_sd_table.colnames = { ...
        'UnitIdx', ...
        'InitialStorage', ...
        'InitialStorageLowerBound', ...
        'InitialStorageUpperBound', ...
        'InitialStorageCost', ...
        'TerminalStoragePrice', ...
        'MinStorageLevel', ...
        'MaxStorageLevel', ...
        };

    ns = numel(ess_rows);
    storage_sd_table.data = zeros(ns, 8);
    storage_sd_table.data(:,1) = ess_rows;   % UnitIdx → gen 行号
    storage_sd_table.data(:,2) = ess_E0;     % 初始能量
    storage_sd_table.data(:,3) = ess_Emin;   % 初始能量下界
    storage_sd_table.data(:,4) = ess_Emax;   % 初始能量上界
    storage_sd_table.data(:,5) = 0;          % 初始能量成本（不收钱）
    storage_sd_table.data(:,6) = 0;          % 终端能量价格（不计残值）
    storage_sd_table.data(:,7) = ess_Emin;   % 运行中的 SOC 下界
    storage_sd_table.data(:,8) = ess_Emax;   % 运行中的 SOC 上界

    % 转成 MOST 使用的 StorageData 结构
    sd = loadstoragedata(storage_sd_table);
end

% MOST 可行性补丁：设置合理的爬坡约束 & VRE 下限清零
% 说明：
%   - 对风电/光伏：仍然允许快速变化（不限制 ramp），只把 PMIN 清零；
%   - 对常规机组（包含 GAS、平衡机组、其它）：按 PMAX 的一定比例给出 RAMP_10 / RAMP_30 / RAMP_AGC，
%     MOST 会根据 Delta_T 自动转换成相邻时段的功率变化约束。

% 确保 gen 矩阵里有 RAMP_* 列
if size(mpc.gen,2) < ID.RAMP_30
    ncol_old = size(mpc.gen,2);
    mpc.gen(:, ncol_old+1:ID.RAMP_30) = 0;
end

ng  = size(mpc.gen,1);
all_idx = (1:ng).';

% 可再生机组（风 + 光）
vre_idx = [idx_wind; idx_pv1; idx_pv2];
vre_idx = vre_idx(vre_idx>=1 & vre_idx<=ng);

% 储能机组（可能有 0/1/2 个）
ess_idx = [];
if exist('idx_ebat_wind','var') && ~isempty(idx_ebat_wind)
    ess_idx = [ess_idx; idx_ebat_wind(:)];
end
if exist('idx_ebat_pv1','var') && ~isempty(idx_ebat_pv1)
    ess_idx = [ess_idx; idx_ebat_pv1(:)];
end
ess_idx = unique(ess_idx);
ess_idx = ess_idx(ess_idx>=1 & ess_idx<=ng);

% 组合出“其它可调热电机组”索引
fixed_idx = unique([vre_idx; idx_gas; idx_slack; idx_grid; ess_idx]);
idx_other_th = setdiff(all_idx, fixed_idx);

% 把这些“Other” 的最小出力放开，让它们可以往下调
if ~isempty(idx_other_th)
    mpc.gen(idx_other_th, ID.PMIN) = 0;
end

% 1) VRE：下限清零，ramp 足够大，相当于不限制
if ~isempty(vre_idx)
    mpc.gen(vre_idx, ID.PMIN) = 0;
    mpc.gen(vre_idx, [ID.RAMP_AGC, ID.RAMP_10, ID.RAMP_30]) = 1e4;
end

% 2) 非 VRE：按 PMAX 的比例设置爬坡约束
non_vre_idx = setdiff(all_idx, vre_idx);
Pmax_all    = mpc.gen(:, ID.PMAX);

% 基本比例（可以按需要调得更紧或更松）
frac_10_base = 0.15;   % 每 10 分钟最多改变 15% PMAX
frac_30_base = 0.30;   % 每 30 分钟最多改变 30% PMAX
frac_agc_base= 0.60;   % 1 小时内大致可以跑满 PMAX

r10  = frac_10_base  * Pmax_all;   % MW/10min
r30  = frac_30_base  * Pmax_all;   % MW/30min
ragc = frac_agc_base * Pmax_all;   % MW/AGC-interval

% 先给所有非 VRE 一个“基准”爬坡
mpc.gen(non_vre_idx, ID.RAMP_10)  = r10(non_vre_idx);
mpc.gen(non_vre_idx, ID.RAMP_30)  = r30(non_vre_idx);
mpc.gen(non_vre_idx, ID.RAMP_AGC) = ragc(non_vre_idx);

% 3) 对燃气机组可以再稍微“慢一点”（更平滑）
if ~isempty(idx_gas)
    idx_gas = idx_gas(idx_gas>=1 & idx_gas<=ng);
    frac_10_g  = 0.15;
    frac_30_g  = 0.35;
    frac_agc_g = 0.80;
    mpc.gen(idx_gas, ID.RAMP_10)  = frac_10_g  * Pmax_all(idx_gas);
    mpc.gen(idx_gas, ID.RAMP_30)  = frac_30_g  * Pmax_all(idx_gas);
    mpc.gen(idx_gas, ID.RAMP_AGC) = frac_agc_g * Pmax_all(idx_gas);
end

% 4) 平衡机组 slack 稍微放宽一点，避免因为太“僵硬”导致不可行
if ~isempty(idx_slack)
    idx_slack = idx_slack(idx_slack>=1 & idx_slack<=ng);
    mpc.gen(idx_slack, ID.RAMP_10)  = 2 * r10(idx_slack);
    mpc.gen(idx_slack, ID.RAMP_30)  = 2 * r30(idx_slack);
    mpc.gen(idx_slack, ID.RAMP_AGC) = 2 * ragc(idx_slack);
end


% 统计打印
fprintf('[Diag] idx_other_th ='); fprintf(' %d', idx_other_th); fprintf('\n');
fprintf('[Diag] ngen(after add) = %d, sum(PMAX) = %.2f MW\n', size(mpc.gen,1), sum(mpc.gen(:,ID.PMAX)));
sw  = shape_wind(T);
sp1 = shape_pv(T);
sp2 = 0.9*shape_pv(T);
fprintf('[Diag] PMAX avail (WIND) min/avg/max = %.2f / %.2f / %.2f MW\n', min(Pwind*sw), mean(Pwind*sw), max(Pwind*sw));
fprintf('[Diag] PMAX avail (PV1 ) min/avg/max = %.2f / %.2f / %.2f MW\n', min(Ppv1*sp1), mean(Ppv1*sp1), max(Ppv1*sp1));
fprintf('[Diag] PMAX avail (PV2 ) min/avg/max = %.2f / %.2f / %.2f MW\n\n', min(Ppv2*sp2), mean(Ppv2*sp2), max(Ppv2*sp2));

%% ------------------- 2) 24h 负荷与电源出力曲线 -------------------
% 目标：构造一个“早晚高/中午低”的电力负荷曲线，同时给风电、光伏、电锅炉
%      生成 24h 的标准化 profile，供 MOST 作为时序扰动使用。
%
% 1) 负荷：先给定一个归一化形状（sum=1），再按 baseMVA 和 Pd 标幺值缩放成 MW
% 2) VRE：wind/pv 的 profile 是 0~1 的系数，乘以机组的 PMAX 得到每小时上限
% 3) 电锅炉：直接写成“每小时固定 MW”的曲线（转成 profile 时落在 CT_TGEN 或 CT_TLOAD）
pmx_wind = Pwind * sw(:)';  pmx_pv1 = Ppv1 * sp1(:)';  pmx_pv2 = Ppv2 * sp2(:)';
setappdata(0,'pmx_wind_cache', pmx_wind);
setappdata(0,'pmx_pv1_cache',  pmx_pv1);
setappdata(0,'pmx_pv2_cache',  pmx_pv2);

profiles = struct('type',{},'table',{},'rows',{},'col',{},'chgtype',{},'values',{});

% (A) 发电机 PMAX(t) —— T×1×3
vals_gen = zeros(T,1,3);
vals_gen(:,1,1) = pmx_wind(:);
vals_gen(:,1,2) = pmx_pv1(:);
vals_gen(:,1,3) = pmx_pv2(:);
profiles(end+1) = struct('type','mpcData','table',CT_TGEN,'rows',[idx_wind; idx_pv1; idx_pv2], ...
    'col',ID.PMAX,'chgtype',CT_REP,'values',vals_gen);

pmx_by_gen = zeros(size(mpc.gen,1), T);
pmx_by_gen([idx_wind; idx_pv1; idx_pv2], :) = squeeze(vals_gen(:,1,:)).';

% (B) 电锅炉 + 早晚高 / 中午低负荷形状（通过电锅炉叠加出明显日负荷曲线）
Pd_base    = sum(mpc.bus(:, ID.PD));   % 原始总有功负荷 (MW)
load_shape = shape_heat(T);            % 早晚高、中午低的标幺曲线 (max = 1)

% ==== 分时电价 / 燃气价标幺系数（24 点形状，T≠24 时插值） ====
lambda_e24 = [ ...
    0.55  %  1  深谷
    0.50  %  2
    0.48  %  3
    0.48  %  4
    0.70  %  5
    0.80  %  6
    0.90  %  7
    1.00  %  8
    1.05  %  9
    1.10  % 10
    1.10  % 11
    1.00  % 12
    0.95  % 13
    0.95  % 14
    1.00  % 15
    1.10  % 16
    1.35  % 17  傍晚尖峰
    1.50  % 18
    1.50  % 19
    1.30  % 20
    0.85  % 21
    0.80  % 22
    0.75  % 23
    0.70];% 24

lambda_g24 = [ ...
    0.90  %  1  夜间略低
    0.88  %  2
    0.87  %  3
    0.87  %  4
    0.88  %  5
    0.90  %  6
    0.95  %  7
    1.00  %  8
    1.02  %  9
    1.05  % 10
    1.08  % 11
    1.10  % 12  白天略高
    1.10  % 13
    1.08  % 14
    1.05  % 15
    1.02  % 16
    1.00  % 17
    1.02  % 18
    1.05  % 19
    1.03  % 20
    0.98  % 21  晚间回落
    0.95  % 22
    0.93  % 23
    0.92];% 24

lambda_e24 = lambda_e24(:);
lambda_g24 = lambda_g24(:);

if T == 24
    lambda_e = lambda_e24;
    lambda_g = lambda_g24;
else
    t24 = (1:24).';
    t   = linspace(1,24,T).';
    lambda_e = interp1(t24, lambda_e24, t, 'pchip');
    lambda_g = interp1(t24, lambda_g24, t, 'pchip');
end


% alpha 控制电锅炉在峰段大概占基准负荷的比例：
%   - 这里取 0.6：峰段电锅炉 ~ 0.6*Pd_base / COP
%   - 叠加到 Pd_base 上后，早晚负荷会比中午明显高一截
alpha = 0.6;
COP   = 1.8;

% 热负荷 & 等效电负荷
heat_th   = alpha * Pd_base * load_shape;   % MW_th, 24h 热负荷
el_boiler = heat_th / COP;                 % MW_el, 等效电锅炉有功

% 按各负荷母线 Pd 权重分摊电锅炉功率
load_buses = find(mpc.bus(:, ID.PD) > 0);
w_share    = mpc.bus(load_buses, ID.PD);
w_share    = w_share / max(1e-9, sum(w_share));

vals_bus = zeros(T,1,numel(load_buses));
for k = 1:numel(load_buses)
    vals_bus(:,1,k) = el_boiler(:) * w_share(k);
end

profiles(end+1) = struct('type','mpcData','table',CT_TLOAD,'rows',load_buses, ...
    'col',CT_LOAD_ALL_P,'chgtype',CT_ADD,'values',vals_bus);


% === 新增：让基准负荷本身也随时间变化（早晚高 / 中午低） ===
load_buses = find(mpc.bus(:, ID.PD) > 0);
Pd_bus0    = mpc.bus(load_buses, ID.PD);   % 每个负荷母线的基准 Pd

% 负荷缩放因子：夜间略低，中午更低，早晚高峰更高
% 这里用已有的 load_shape 做个线性变换：min≈0.7, max≈1.3
scale = 0.7 + 0.6 * load_shape(:);   % T×1，早晚 ~1.3，中午 ~0.7–0.8

vals_bus2 = zeros(T,1,numel(load_buses));
for k = 1:numel(load_buses)
    % 在基准负荷的基础上加一个增量：(scale(t)-1) * Pd_bus0(k)
    for tt = 1:T
        vals_bus2(tt,1,k) = (scale(tt) - 1.0) * Pd_bus0(k);
    end
end

% 叠加到原始 Pd 上（CT_ADD 表示在基准值上加增量）
profiles(end+1) = struct('type','mpcData','table',CT_TLOAD,'rows',load_buses, ...
    'col',CT_LOAD_ALL_P,'chgtype',CT_ADD,'values',vals_bus2);

% ==== 分时电价 / 燃气价：通过 CT_TGENCOST 修改线性成本斜率 ====
% 注意：这里假定前面已经用 pad_gencost_linear 把 idx_grid / idx_gas
%       的成本设置为常数边际成本 base_grid_cost / base_gas_cost。

base_grid_cost = 60;   % 与上文 mpc.gencost(idx_grid,:) 一致
base_gas_cost  = 45;   % 与上文 mpc.gencost(idx_gas,:) 一致

% --- GRID 机组：电价随时间变化 ---
vals_gc_grid = zeros(T,1,1);
for tt = 1:T
    vals_gc_grid(tt,1,1) = base_grid_cost * lambda_e(tt);
end

profiles(end+1) = struct( ...
    'type',    'mpcData', ...
    'table',   CT_TGENCOST, ...
    'rows',    idx_grid, ...       % 主网购电机组
    'col',     CT_MODCOST, ...
    'chgtype', CT_REP, ...         % 直接用新的边际成本替换
    'values',  vals_gc_grid );

% === “Other” 常规机组：成本介于 GAS 和 GRID 之间，比如 50 $/MWh ===
if ~isempty(idx_other_th)
    base_other_cost = 50;           % 45 < 50 < 60

    n_other = numel(idx_other_th);
    vals_gc_other = zeros(T, 1, n_other);
    for k = 1:n_other
        vals_gc_other(:,1,k) = base_other_cost * lambda_e(:);
    end

    profiles(end+1) = struct( ...
        'type',    'mpcData', ...
        'table',   CT_TGENCOST, ...
        'rows',    idx_other_th, ...
        'col',     CT_MODCOST, ...
        'chgtype', CT_REP, ...
        'values',  vals_gc_other );
end

% --- GAS 机组：燃气价更平缓一些 ---
vals_gc_gas = zeros(T,1,1);
for tt = 1:T
    vals_gc_gas(tt,1,1) = base_gas_cost * lambda_g(tt);
end

profiles(end+1) = struct( ...
    'type',    'mpcData', ...
    'table',   CT_TGENCOST, ...
    'rows',    idx_gas, ...        % 燃气机组
    'col',     CT_MODCOST, ...
    'chgtype', CT_REP, ...
    'values',  vals_gc_gas );

fprintf('GRID cost(t) ='); fprintf(' %.1f', base_grid_cost * lambda_e); fprintf('\n');
fprintf('GAS  cost(t) ='); fprintf(' %.1f', base_gas_cost  * lambda_g); fprintf('\n');

%% ------------------- 3) 供需余量摘要 -------------------
Pgt_eff = mpc.gen(idx_gas, ID.PMAX);
supply_wo_slack = pmx_wind + pmx_pv1 + pmx_pv2 + Pgt_eff;
demand_base = sum(mpc.bus(:,ID.PD)) + el_boiler.';  % 含电锅炉
slack_pmax = mpc.gen(idx_slack, ID.PMAX);
diff1 = supply_wo_slack - demand_base;
diff2 = supply_wo_slack + slack_pmax - demand_base;
fprintf('SupplyMax - Demand (no-slack)  min/avg/max = %.2f / %.2f / %.2f MW\n', min(diff1), mean(diff1), max(diff1));
fprintf('SupplyMax - Demand (with-slack) min/avg/max = %.2f / %.2f / %.2f MW\n\n', min(diff2), mean(diff2), max(diff2));

%% --------- 4) DC-OPF 预检：在跑 MOST 之前做一次静态 DC OPF ---------
% 目的：防止 mpc 数据本身就不可行（比如 PMAX 太小、机组全部停机），
%      先用 rundcopf 做一次简单 DC OPF，打印总成本和功率平衡。
try
    m2mpc = mpc;
    if size(m2mpc.branch,2) >= ID.RATE_A, m2mpc.branch(:, ID.RATE_A) = 0; end
    m2mpc.gencost = enforce_linear_cost(m2mpc.gencost, size(m2mpc.gen,1), 20);
    mpopt_dc = mpoption('opf.dc.solver','MIPS','verbose',2);
    fprintf('[DC-OPF 预检] 输入: ngen=%d, sum(PMAX)=%.3f MW, sum(Pd)=%.3f MW\n', size(m2mpc.gen,1), sum(m2mpc.gen(:,ID.PMAX)), sum(m2mpc.bus(:,ID.PD)));
    r0 = rundcopf(m2mpc, mpopt_dc);
    Pg0 = sum(r0.gen(:, ID.PG));
    if size(r0.gen,1) ~= size(m2mpc.gen,1)
        warning('[DC-OPF 预检] 机组数不一致，检查 GEN_STATUS/PMAX。');
    end
    fprintf('[DC-OPF 预检] 成功：Objective = %.3f $/h，Total Pg = %.3f MW，Total Pd = %.3f MW\n\n', r0.f, Pg0, sum(m2mpc.bus(:,ID.PD)));
catch ME
    warning('DC-OPF 预检查失败（%s）。', ME.message);
end

%% ------------------- 5) MOST 求解 -------------------
ok_most = false; resE = struct(); mpc_used = mpc; mdE = []; profiles_ok = false;
LOAD_SCALE_LIST = [1.00, 0.80, 0.60];

for k = 1:length(LOAD_SCALE_LIST)
    scl = LOAD_SCALE_LIST(k);
    try
        m_try = mpc; m_try.bus(:, ID.PD) = m_try.bus(:, ID.PD) * scl;

        try
            mdE = try_loadmd_multi(m_try, T, profiles, xgd, sd);
            profiles_ok = true;
        catch ME1
            warn_slim('[MOST] 首次失败：%s', ME1.message);
            mdE = try_loadmd_multi(m_try, T, [], xgd, sd);
            profiles_ok = false;
        end

        try
            [~, order] = ext2int(m_try);
            vre_rows_int = order.gen.e2i([idx_wind; idx_pv1; idx_pv2]);
            vre_rows_int = vre_rows_int(vre_rows_int > 0);
        catch
            vre_rows_int = [idx_wind; idx_pv1; idx_pv2];
        end
        setappdata(0,'vre_rows_int_tmp',vre_rows_int);

        resE = most(mdE, mpopt_most);
        if isfield(resE,'results') && isfield(resE.results,'success') && resE.results.success
            ok_most = true; mpc_used = m_try; break;
        else
            if k < numel(LOAD_SCALE_LIST)
                fprintf('MOST 初次失败，回退：统一缩放基础负荷到 %.2f ...\n', LOAD_SCALE_LIST(k+1));
            end
        end
    catch ME
        fprintf('[MOST-ERR] %s\n', ME.getReport('basic'));
        if k < numel(LOAD_SCALE_LIST)
            fprintf('MOST 初次失败，回退：统一缩放基础负荷到 %.2f ...\n', LOAD_SCALE_LIST(k+1));
        end
    end
end

if ~ok_most
    warning('MOST 未求得可行解或失败。将绘制占位曲线辅助诊断。');
else
    idx_pos = mpc.bus(:, ID.PD) > 0;
    scl_used = mean( mpc_used.bus(idx_pos, ID.PD) ./ max(1e-9, mpc.bus(idx_pos, ID.PD)) );
    fprintf('[MOST] load scale used = %.2f\n', scl_used);
    fprintf('[MOST] sum(el_boiler) = %.3f MWh\n', sum(el_boiler));
end

%% ------------------- 6) 调用 MPNG 进行气网潮流 + 压缩机电耗提取 -------------------
% 这一部分做三件事：
%  (1) 按多种“候选 payload” 尝试调用 mpng()，兼容不同版本的接口：
%      - 官方推荐签名：mpng(mpgc)，其中 mpgc.matgas = mgc_case, mpgc.mpc = mpc_used
%      - 极简签名：mpng(struct('matgas',mgc_case,'mpc',mpc_used,...))
%  (2) 调用成功后，打印 48 节点气网清单（print_gas_inventory），方便检查管道/压缩机/气源配置
%  (3) 根据 opts.comp_pc_unit 判定压缩机电功率的口径（MW 或 MWh/day），
%      将各压缩机在 24h 的电耗组成 comp_el(i,t) 矩阵，供 6.5) 步骤反馈给 MOST
comp_el = [];        % #comp × 24（MW）；失败则置零
eg = []; mpng_ok = false; mpng_try = 0;

% 1) 读取 48 节点气网
mgc_case = [];
try
    if exist('ng_case48','file')==2
        mgc_case = ng_case48();
    elseif exist('ng_case48_pu','file')==2
        mgc_case = ng_case48_pu();
    else
        error('未找到 48 节点气网案例（ng_case48.m / ng_case48_pu.m）。');
    end
catch ME
    warning('[GAS] 读取气网失败：%s', ME.message);
end

% 2) 组装 connect（最小可用映射；GEN_ID=燃气机、NODE_ID=10、EFF=0.007 MMSCFD/MW 可按需改）
connect = struct();
connect.version = 1;
connect.power = struct();
connect.power.time = ones(1, T);      % 24 个 1h
% demands 将在 normalize_mpng_case() 中自动由 mpc_used.bus(:,PD/QD) 复制为 NB×T

% —— 电-气映射：燃气机作为耗气负荷 ——
define_constants_gas;
try NB_g = detect_gas_nodes(mgc_case); catch, NB_g = 0; end
GEN_ID = 1; NODE_ID = 2; EFF = 3;            % 仅本脚本使用
node_default = max(1, min(10, NB_g));        % 默认连到气网节点 #10
eff_default  = 7e-3;                         % 0.007 MMSCFD/MW
connect.interc = struct();
connect.interc.term = [idx_gas, node_default, eff_default];   % 一条燃气机-气节点映射

% —— 电驱压缩机接到“风机同一母线”（可改到任意母线）——
elec_comp_ids = []; % 记录最终按电驱接入的压缩机行号
BUS_ID_FOR_COMP = mpc.gen(idx_wind, ID.GEN_BUS);  % 缺省接到风电母线
if isnumeric(opts.comp_bus)
    BUS_ID_FOR_COMP = opts.comp_bus;
end

has_comp = isfield(mgc_case,'comp') && ~isempty(mgc_case.comp);
if has_comp
    C = mgc_case.comp; define_constants_gas;     % 需要 MPNG 的 gas 常量
    define_constants_gas;                 % 定义 TYPE_C, COMP_P, COMP_G, PC_C 等
    isP = (C(:,TYPE_C) == COMP_P);               % 原生电驱
    % 1) 选择压缩机集合
    if (ischar(opts.comp_ids) && strcmpi(opts.comp_ids,'all')) ...
            || (isstring(opts.comp_ids) && opts.comp_ids=="all")
        pick = (1:size(C,1)).';
    else
        pick = opts.comp_ids(:);
        pick = pick(pick>=1 & pick<=size(C,1));
        if isempty(pick), pick = (1:size(C,1)).'; end
    end

    % 2) 默认只认原生电驱
    ids = pick(isP(pick));

    % 3) 若开启 force_comp_as_el，则把选中的“气驱”在 mgc_case 里也改成电驱
    if opts.force_comp_as_el
        ids = pick;
        nonP = ids(~isP(ids));   % 这些是原本的 COMP_G
        if ~isempty(nonP)
            warning('[MPNG] 将 %d 台非电驱压缩机按电驱接入电网。', numel(nonP));

            % ★ 核心：同步修改气网模型，把这些行标记为 COMP_P ★
            mgc_case.comp(nonP, TYPE_C) = COMP_P;
            % 同步 isP，避免后续代码再用到 isP 时不一致
            isP(nonP) = true;
        end
    end

    % 4) 生成 connect.interc.comp 映射
    if ~isempty(ids)
        connect.interc.comp = [ids(:), repmat(BUS_ID_FOR_COMP, numel(ids), 1)];
        elec_comp_ids = ids(:);
    else
        connect.interc.comp = [];
    end
else
    connect.interc.comp = [];
end


% 兼容不同版本 MPNG：既在 power 下，也在顶层提供 UC/ramp/energy/sr/cost 字段
connect.power.UC        = [];
connect.power.ramp_time = [];
connect.power.energy    = [];
connect.power.sr        = [];
connect.power.cost      = 0;
connect.UC        = [];
connect.ramp_time = [];
connect.energy    = [];
connect.sr        = [];
connect.cost      = 0;

% 3) 构建 mpgc 候选载荷并尝试调用 mpng（单参数优先）
cands = {};
try
    cands = build_mpng_payloads(mpc_used, mgc_case, connect, ID, T);
    fprintf('[MPNG] candidates kept = %d\n', numel(cands));
catch ME
    warning('[MPNG] 构造候选载荷失败：%s', ME.message);
end

reasons = {};
for i = 1:numel(cands)
    mpgc_try = cands{i};
    mpgc_try = enforce_mpng_top_fields(mpgc_try, mpc_used, mgc_case, connect);
    mpng_try = mpng_try + 1;
    fprintf('[MPNG-TRY#%d] gen=%dx%d bus=%dx%d branch=%dx%d\n', mpng_try, size(mpgc_try.gen), size(mpgc_try.bus), size(mpgc_try.branch));
    % 只试“单结构体”签名，其次试“单结构体 + mpopt”（若第一种返回 success=false）
    try
        eg = mpng(mpgc_try);   % 单参，避免 Too many input arguments
        if isfield(eg,'success') && eg.success
            mpng_ok = true; fprintf('[MPNG] 使用 AC-OPF 成功（单参）。\n'); break;
        else
            % 若返回不成功，尝试带 mpopt（有些版本支持第二参）
            try
                eg = mpng(mpgc_try, mpopt_mpng_AC);
                if isfield(eg,'success') && eg.success
                    mpng_ok = true; fprintf('[MPNG] 使用 AC-OPF (MIPS) 成功（struct+mpopt）。\n'); break;
                end
            catch ME2
                reasons{end+1} = sprintf('AC fail(2-arg): %s', ME2.message);
            end
        end
    catch ME1
        reasons{end+1} = sprintf('AC fail(1-arg): %s', ME1.message);
        % 尝试 IPOPT（仍旧优先单参）
        try
            mpgc_try_ip = mpgc_try;
            mpgc_try_ip.mpopt = mpopt_mpng_IPOPT; % 某些版本读 mpgc.mpopt
            eg = mpng(mpgc_try_ip);
            if isfield(eg,'success') && eg.success
                mpng_ok = true; fprintf('[MPNG] 使用 AC-OPF (IPOPT) 成功（单参+mpgc.mpopt）。\n'); break;
            end
        catch ME3
            reasons{end+1} = sprintf('IPOPT fail(1-arg): %s', ME3.message);
        end
        % 最后尝试 DC（单参）
        try
            mpgc_try_dc = mpgc_try;
            mpgc_try_dc.mpopt = mpopt_mpng_DC;
            eg = mpng(mpgc_try_dc);
            if isfield(eg,'success') && eg.success
                mpng_ok = true; fprintf('[MPNG] 使用 DC-OPF 成功（单参+mpgc.mpopt）。\n'); break;
            end
        catch ME4
            reasons{end+1} = sprintf('DC fail(1-arg): %s', ME4.message);
        end
    end
end

if ~mpng_ok && ~isempty(reasons)
    fprintf('[MPNG] 全部签名均未成功，累计原因如下：\n');
    for k = 1:numel(reasons)
        fprintf('  - %s\n', reasons{k});
    end
end

% 4) 气网参数与连接性自检打印 + “48节点气网清单” + 提取压缩机电耗 comp_el
define_constants_gas
if ~isempty(mgc_case)
    try
        fprintf('[GAS] 基值: fbase=%.3f MMSCFD, pbase=%.1f psi, wbase=%.3f MW\n', ...
            safe_num(mgc_case,'fbase',1), safe_num(mgc_case,'pbase',1), safe_num(mgc_case,'wbase',1));
        nn = 0; nw = 0; no = 0; nc = 0; ns = 0;
        if isfield(mgc_case,'node') && isfield(mgc_case.node,'info') && ~isempty(mgc_case.node.info), nn = size(mgc_case.node.info,1); end
        if isfield(mgc_case,'well')   && ~isempty(mgc_case.well),   nw = size(mgc_case.well,1); end
        if isfield(mgc_case,'pipe')   && ~isempty(mgc_case.pipe),   no = size(mgc_case.pipe,1); end
        if isfield(mgc_case,'comp')   && ~isempty(mgc_case.comp),   nc = size(mgc_case.comp,1); end
        if isfield(mgc_case,'sto')    && ~isempty(mgc_case.sto),    ns = size(mgc_case.sto,1); end
        fprintf('[GAS] 规模: nodes=%d, wells=%d, pipes=%d, comps=%d, storage=%d\n', nn, nw, no, nc, ns);
        if nc>0
            r = mgc_case.comp(:,RATIO_C);
            nP = nnz(mgc_case.comp(:,TYPE_C)==COMP_P);
            nG = nnz(mgc_case.comp(:,TYPE_C)==COMP_G);
            fprintf('[GAS] 压缩机: COMP_P=%d, COMP_G=%d, ratio[min/avg/max]=%.3f/%.3f/%.3f\n', nP, nG, min(r), mean(r), max(r));
        end
        if ~isempty(connect.interc.term)
            fprintf('[LINK] interc.term = [GEN_ID=%d, NODE_ID=%d, EFF=%.4g MMSCFD/MW]\n', ...
                connect.interc.term(1), connect.interc.term(2), connect.interc.term(3));
        else
            fprintf('[LINK] interc.term 未设置（燃气机未与气网耦合）。\n');
        end

        % === 打印“48 节点气网清单” ===
        print_gas_inventory(mgc_case);

    catch MEi
        warning('气网信息打印失败：%s', MEi.message);
    end
end

% 提取压缩机功率（MW×24）
comp_el = zeros(0, T);
if mpng_ok && isstruct(eg)
    try
        % 优先：psi_p（常见于电驱）——直接转 MW
        if isfield(eg,'var') && isfield(eg.var,'val') && isfield(eg.var.val,'psi_p') && ~isempty(eg.var.val.psi_p)
            psi = double(eg.var.val.psi_p(:));
            wb  = safe_num(eg,'mgc.wbase',safe_num(eg,'baseMVA',mpc_used.baseMVA));
            psi_MW = psi * wb;
            ncomp = size(eg.mgc.comp,1); comp_el = zeros(ncomp, T);
            if ~isempty(elec_comp_ids)
                m = min(numel(psi_MW), numel(elec_comp_ids));
                comp_el(elec_comp_ids(1:m), :) = repmat(psi_MW(1:m), 1, T);
            else
                define_constants_gas; iscomp_p = (eg.mgc.comp(:,TYPE_C)==COMP_P);
                comp_el(iscomp_p, :) = repmat(psi_MW(:), 1, T);
            end
            fprintf('[MPNG] 压缩机电耗由 psi_p 推得：24h 合计 %.3f MWh\n', sum(comp_el(:)));

            % 其次：读取 mgc.comp(:,PC_C)
        elseif isfield(eg,'mgc') && isfield(eg.mgc,'comp') && size(eg.mgc.comp,2)>=PC_C
            Pc_raw = double(eg.mgc.comp(:,PC_C));   % 口径可能是 MW 或 MWh/day
            ncomp  = size(eg.mgc.comp,1);
            comp_el = zeros(ncomp, T);

            rows = elec_comp_ids(:);
            if isempty(rows), rows = find(eg.mgc.comp(:,PC_C) > 0); end
            if isempty(rows)
                define_constants_gas; rows = find(eg.mgc.comp(:,TYPE_C) == COMP_P);
            end

            day_hours = safe_num(eg,'connect.day_hours',safe_num(eg,'connect.power.day_hours', T));
            if ~isfinite(day_hours) || day_hours <= 0, day_hours = T; end

            % —— 单位判别：opts.comp_pc_unit = 'MW'|'MWh/day'|'auto'
            unit_used = 'MW'; Pc_MW = Pc_raw;
            switch lower(string(opts.comp_pc_unit))
                case "mwh/day"
                    Pc_MW = Pc_raw / day_hours; unit_used = 'MWh/day';
                case "mw"
                    Pc_MW = Pc_raw;            unit_used = 'MW';
                otherwise % auto
                    if max(Pc_raw) > 10 && max(Pc_raw)/day_hours < 10
                        Pc_MW = Pc_raw / day_hours; unit_used = 'MWh/day';
                    else
                        Pc_MW = Pc_raw;            unit_used = 'MW';
                    end
            end

            if ~isempty(rows), comp_el(rows, :) = repmat(Pc_MW(rows), 1, T); end
            E24 = sum(comp_el(:));  % 注意：此处已经是 MWh，不要再 *24
            fprintf('[MPNG] 压缩机电耗由 mgc.comp(:,PC_C) 读取（口径=%s）：24h 合计 %.3f MWh\n', unit_used, E24);
        else
            fprintf('[MPNG] 结果中无 psi_p / PC_C，压缩机电耗置零。\n');
            comp_el = zeros(0,T);
        end
    catch MEp
        warning('压缩机电耗提取失败：%s；置零。', MEp.message);
        comp_el = zeros(0, T);
    end
end

%% ------------------- 6.5) 强耦合迭代：将压缩机电耗反馈给 MOST -------------------
% 思路：
%   - 第一次 MOST 是“不考虑压缩机电耗”的电网优化；
%   - MPNG 求解后，我们拿到每台压缩机的电功率/电耗 comp_el(i,t)；
%   - connect.interc.comp = [COMP_ID, BUS_ID] 给出了“压缩机 → 电网母线”的映射；
%   - 把这些电耗等效为对应母线上的“附加固定负荷”（CT_TLOAD / CT_LOAD_ALL_P）；
%   - 重新构造 profiles2，再跑一遍 MOST（即 MOST-2），形成真正电-气强耦合的调度结果。
%
% 为了避免 MOST 再次不收敛，这里沿用了 5) 中的 LOAD_SCALE_LIST 降级逻辑。

if ok_most && ~isempty(comp_el)
    NB = size(mpc.bus, 1);
    add_load_MW = zeros(NB, 1);   % 每个电网母线上，由压缩机带来的额外负荷功率（MW）

    try
        if isfield(connect, 'interc') && isfield(connect.interc, 'comp') && ~isempty(connect.interc.comp)
            % connect.interc.comp: [COMP_ID, BUS_ID]
            map = connect.interc.comp;
            nmap = size(map, 1);

            % 决定 bus 号所在列：若 ID.BUS_I 存在就用它，否则用第1列
            if exist('ID', 'var') && isstruct(ID) && isfield(ID, 'BUS_I')
                col_bus = ID.BUS_I;
            else
                col_bus = 1;   % MATPOWER 标准：bus(:,1) 即 BUS_I
            end

            for kk = 1:nmap
                cid = map(kk, 1);     % 压缩机行号（mgc.comp 的行）
                bid = map(kk, 2);     % 电网母线号（mpc.bus 的 BUS_I）
                % 容错：索引越界就跳过
                if cid >= 1 && cid <= size(comp_el, 1)
                    Pcid = comp_el(cid, 1);  % MW，当前代码中24h恒定
                    % 找到这个 BUS_ID 在 mpc.bus 中对应的行号
                    bus_row = find(mpc.bus(:, col_bus) == bid, 1);
                    if ~isempty(bus_row)
                        add_load_MW(bus_row) = add_load_MW(bus_row) + Pcid;
                    end
                end
            end
        end
    catch ME_map
        warning('[COUPLED] 压缩机负荷-母线映射失败：%s，跳过强耦合迭代。', ME_map.message);
        add_load_MW(:) = 0;
    end


    if any(add_load_MW > 1e-6)
        fprintf('[COUPLED] 将压缩机电耗作为固定负荷反馈到 MOST，总功率 = %.3f MW\n', sum(add_load_MW));

        % 1) 基于 add_load_MW 构造一个新的负荷 profile（CT_TLOAD / CT_LOAD_ALL_P）
        rows_comp = find(add_load_MW > 1e-6);      % 只给有压缩机负荷的母线加 profile
        vals_comp = zeros(T, 1, numel(rows_comp)); % T×1×N_bus
        for kk = 1:numel(rows_comp)
            b = rows_comp(kk);
            vals_comp(:, 1, kk) = add_load_MW(b);  % 每个小时都是相同的 MW
        end

        profiles2 = profiles;
        profiles2(end+1) = struct( ...
            'type',   'mpcData', ...
            'table',  CT_TLOAD, ...
            'rows',   rows_comp, ...
            'col',    CT_LOAD_ALL_P, ...
            'chgtype',CT_ADD, ...
            'values', vals_comp);

        % 2) 带压缩机负荷重新跑 MOST（沿用原来的 LOAD_SCALE_LIST 策略）
        ok_most2     = false;
        resE2        = struct();
        mpc_used2    = mpc;
        mdE2         = [];
        profiles_ok2 = false;

        for kk = 1:length(LOAD_SCALE_LIST)
            scl2 = LOAD_SCALE_LIST(kk);
            try
                m2 = mpc;
                % 仍然统一缩放基准负荷，然后在 profiles2 中叠加压缩机负荷
                m2.bus(:, ID.PD) = m2.bus(:, ID.PD) * scl2;

                try
                    mdE2 = try_loadmd_multi(m2, T, profiles2, xgd, sd);
                    profiles_ok2 = true;
                catch ME2a
                    warn_slim('[MOST-2] 首次失败：%s', ME2a.message);
                    mdE2 = try_loadmd_multi(m2, T, [], xgd, sd);
                    profiles_ok2 = false;
                end

                resE2 = most(mdE2, mpopt_most);
                if isfield(resE2, 'results') && isfield(resE2.results, 'success') && resE2.results.success
                    ok_most2  = true;
                    mpc_used2 = m2;
                    break;
                else
                    if kk < numel(LOAD_SCALE_LIST)
                        fprintf('MOST-2 初次失败，回退：统一缩放基础负荷到 %.2f ...\n', LOAD_SCALE_LIST(kk+1));
                    end
                end
            catch ME2b
                fprintf('[MOST-2-ERR] %s\n', ME2b.getReport('basic'));
                if kk < numel(LOAD_SCALE_LIST)
                    fprintf('MOST-2 失败，尝试缩放基础负荷到 %.2f ...\n', LOAD_SCALE_LIST(kk+1));
                end
            end
        end

        if ok_most2
            % 用“强耦合”结果覆盖第一次 MOST 的结果
            ok_most     = ok_most2;
            resE        = resE2;
            mdE         = mdE2;
            mpc_used    = mpc_used2;
            profiles_ok = profiles_ok2;

            % 重新计算负荷缩放系数（只是为了打印）
            idx_pos  = mpc.bus(:, ID.PD) > 0;
            scl_used = mean( mpc_used.bus(idx_pos, ID.PD) ./ max(1e-9, mpc.bus(idx_pos, ID.PD)) );
            fprintf('[MOST-2] load scale used (with compressor load) = %.2f\n', scl_used);
        else
            fprintf('[MOST-2] 未能在压缩机负荷反馈下找到可行解，保留第一次 MOST 结果。\n');
        end
    else
        fprintf('[COUPLED] 压缩机电耗为零或未识别，跳过强耦合迭代。\n');
    end
end


%% ------------------- 7) 评估与打印 -------------------
% 这一部分把 MOST 的原始结果转成更直观的 KPI：
%   1) 从 resE.results 中提取总成本 cost_total
%   2) 通过 Pg2 计算 24h 总发电量 E_gen_MWh（近似等于负荷电量）
%   3) 对风/光机组，计算：
%        - 可用上限能量 ren_avail_MWh（由 pmx_by_gen）
%        - 实际调度能量 ren_sched_MWh
%        - 弃风/弃光电量 curtail_MWh
%        - 可再生利用率 ren_util = ren_sched_MWh / ren_avail_MWh
%   4) 按“成本越低越好、弃电越少越好”的习惯，把 obj_econ/obj_curtail 取负号，
%      以便在 DRL 里做“最大化”问题。
[obj_econ, obj_curtail, obj_smooth] = deal(0,0,0);

% 1) 总成本：只取 MOST 目标，避免双计
cost_total = NaN;
if isstruct(resE) && isfield(resE,'results')
    R = resE.results;
    if isfield(R, 'f') && isscalar(R.f)
        cost_total = double(R.f);
    elseif isfield(R,'Outputs') && isfield(R.Outputs,'Obj') && ~isempty(R.Outputs.Obj)
        cost_total = sum(double(R.Outputs.Obj(:)));
    end
end

% 2) 取发电功率矩阵 Pg2（NG×T，内部顺序）
Pg2 = take_Pg(resE);
if ~isempty(Pg2) && size(Pg2,2) ~= T && size(Pg2,1) == T, Pg2 = Pg2.'; end

% === 新增：从 Pg2 中提取储能出力时间序列 ===
ts_ebat_wind = [];
ts_ebat_pv1  = [];

if ~isempty(Pg2)
    % 风电侧储能：idx_ebat_wind 在上文“储能挂接”处已经初始化为 []，
    % 若确实配置了 wind 储能并且 Pmax>0，则这里会是一个合法 gen 行号
    if exist('idx_ebat_wind','var') && ~isempty(idx_ebat_wind)
        ts_ebat_wind = take_ts(Pg2, mpc_used, idx_ebat_wind);  % 1×T
    end

    % PV1 侧储能
    if exist('idx_ebat_pv1','var') && ~isempty(idx_ebat_pv1)
        ts_ebat_pv1  = take_ts(Pg2, mpc_used, idx_ebat_pv1);   % 1×T
    end
end

% === 由储能出力重建能量 En 和 SOC（不依赖 resE.stor） ===
SOC_storage_wind = [];
SOC_storage_pv1  = [];
En_storage_wind  = [];
En_storage_pv1   = [];


% 单位时间步长（小时）
dt = 1;
if exist('mdE','var') && isfield(mdE,'Delta_T') && ~isempty(mdE.Delta_T)
    dt = mdE.Delta_T;
end

% 小工具：给定功率序列 P(1..T)、初始能量 E0、上限 Emax，积分出 E(1..T)
% 约定：P>0 放电，P<0 充电；eta_ch / eta_dis 在前文“储能挂接”处定义
reconstruct_E = @(P, E0, Emax) local_reconstruct_E(P, E0, Emax, eta_ch, eta_dis, dt, T);

% --- wind 侧储能 ---
if ~isempty(ts_ebat_wind) && exist('ess_rows','var') && ~isempty(ess_rows) ...
        && exist('idx_ebat_wind','var') && ~isempty(idx_ebat_wind)

    pos_w = find(ess_rows == idx_ebat_wind, 1);
    if ~isempty(pos_w)
        Emax_w = ess_Emax(pos_w);
        E0_w   = ess_E0(pos_w);

        En_storage_wind = reconstruct_E(ts_ebat_wind(:).', E0_w, Emax_w);
        if Emax_w > 0
            SOC_storage_wind = En_storage_wind ./ Emax_w;
        end
    end
end

% --- PV1 侧储能 ---
if ~isempty(ts_ebat_pv1) && exist('ess_rows','var') && ~isempty(ess_rows) ...
        && exist('idx_ebat_pv1','var') && ~isempty(idx_ebat_pv1)

    pos_p = find(ess_rows == idx_ebat_pv1, 1);
    if ~isempty(pos_p)
        Emax_p = ess_Emax(pos_p);
        E0_p   = ess_E0(pos_p);

        En_storage_pv1 = reconstruct_E(ts_ebat_pv1(:).', E0_p, Emax_p);
        if Emax_p > 0
            SOC_storage_pv1 = En_storage_pv1 ./ Emax_p;
        end
    end
end


if ~isempty(Pg2)
    % 风电侧储能：idx_ebat_wind 在上文“储能挂接”处已经初始化为 []，
    % 若确实配置了 wind 储能并且 Pmax>0，则这里会是一个合法 gen 行号
    if exist('idx_ebat_wind','var') && ~isempty(idx_ebat_wind)
        ts_ebat_wind = take_ts(Pg2, mpc_used, idx_ebat_wind);  % 1×T
    end

    % PV1 侧储能
    if exist('idx_ebat_pv1','var') && ~isempty(idx_ebat_pv1)
        ts_ebat_pv1  = take_ts(Pg2, mpc_used, idx_ebat_pv1);   % 1×T
    end
end



% 3) 基线负荷与总发电量（MWh）
E_gen_MWh  = NaN;
if ~isempty(Pg2), E_gen_MWh = sum(Pg2(:)); end

% 4) 电锅炉状态文本
inc_cmd = sum(el_boiler);
if ok_most
    boiler_report_txt = sprintf('[APPLIED] 电锅炉按负荷曲线注入 MOST：seen=%.3f MWh（等同于 cmd=%.3f）', inc_cmd, inc_cmd);
else
    boiler_report_txt = sprintf('[SKIP] MOST 未求解成功，电锅炉曲线仅用于预检统计：cmd=%.3f MWh', inc_cmd);
end

% 5) 可再生统计
ren_avail_MWh = sum(pmx_by_gen([idx_wind; idx_pv1; idx_pv2], :), "all");
ren_sched_MWh = NaN; curtail_MWh = NaN; therm_sched_MWh = NaN; ren_util = NaN;

vre_msg = '';
if ~isempty(Pg2)
    vre_rows_int = getappdata(0,'vre_rows_int_tmp');
    if isempty(vre_rows_int) || any(vre_rows_int<1) || any(vre_rows_int>size(Pg2,1))
        try
            [~, order2] = ext2int(mpc_used);
            vre_rows_int = order2.gen.e2i([idx_wind; idx_pv1; idx_pv2]);
            vre_rows_int = vre_rows_int(vre_rows_int > 0);
        catch
            vre_rows_int = [idx_wind; idx_pv1; idx_pv2];
        end
    end

    Pg_vre = safe_pick_rows(Pg2, vre_rows_int, [idx_wind; idx_pv1; idx_pv2]);   % 3×T
    Av_vre = pmx_by_gen([idx_wind; idx_pv1; idx_pv2], :);                       % 3×T

    % 越界检测（裁剪前）
    over_raw = Pg_vre > (Av_vre + 1e-6);
    if any(over_raw(:)), vre_msg = '【提示】检测到 VRE 调度超过可用上限，已按上限裁剪统计（请检查 PMAX(t) profile 是否生效）。'; end

    Pg_vre_clip     = min(Pg_vre, Av_vre);
    ren_sched_MWh   = sum(Pg_vre_clip(:));
    curtail_MWh     = max(0, ren_avail_MWh - ren_sched_MWh);
    therm_sched_MWh = sum(Pg2(:)) - ren_sched_MWh;
    ren_util        = ren_sched_MWh / max(1e-9, ren_avail_MWh);
end

%% ================== 目标4：电压偏差（AC PF 后处理） ==================
% 说明：
% - MOST 采用 DC OPF，不产生电压幅值 VM，因此这里用 AC 潮流 runpf 做“后处理评估”。
% - 每个小时用 Pg(t) + Pd(t) 构造一个 mpc_hour，然后 runpf 得到 VM(t)。
% - 指标建议：24h 平均电压绝对偏差 avg(|VM-1|)（越小越好）。
%
% 超参数（可用 opts.vdev 覆盖）
vdev_fail_pen = 0.20;        % 潮流失败时的单小时惩罚（pu 量级）
vref = 1.0;                  % 参考电压
if isfield(opts,'vdev') && isstruct(opts.vdev)
    if isfield(opts.vdev,'fail_pen') && isfinite(opts.vdev.fail_pen), vdev_fail_pen = opts.vdev.fail_pen; end
    if isfield(opts.vdev,'vref')     && isfinite(opts.vdev.vref),     vref = opts.vdev.vref; end
end

% vdev 评估方式：默认 ACPF；可选 'acopf'（优先）→失败回退 ACPF →仍失败用 fail_pen
vdev_eval = 'acpf';
vdev_fallback_to_pf = true;
if isfield(opts,'vdev_eval') && ~isempty(opts.vdev_eval)
    vdev_eval = opts.vdev_eval;
end
if isfield(opts,'vdev') && isstruct(opts.vdev)
    if isfield(opts.vdev,'eval') && ~isempty(opts.vdev.eval)
        vdev_eval = opts.vdev.eval;
    end
    if isfield(opts.vdev,'fallback_to_pf') && ~isempty(opts.vdev.fallback_to_pf)
        vdev_fallback_to_pf = logical(opts.vdev.fallback_to_pf);
    end
end

% --- vdev 评估：优先使用 MPNG 的 AC-OPF 结果（若可用）；否则按 acopf/acpf 计算 ---
use_mpng_v = strcmpi(vdev_eval,'mpng') || strcmpi(vdev_eval,'acopf');  % acopf 时也先尝试用 mpng（更稳）
used_mpng_v = false;
if use_mpng_v && mpng_ok && isstruct(eg)
    try
        nb_base = size(mpc_used.bus, 1);
        [obj_vdev, vdev_kpi] = calc_voltage_deviation_from_mpng(eg, ID, T, vref, vdev_fail_pen, nb_base);
        used_mpng_v = isfield(vdev_kpi,'used') && logical(vdev_kpi.used);
    catch
        used_mpng_v = false;
    end
end

if ~used_mpng_v
    if strcmpi(vdev_eval,'acopf')
        [obj_vdev, vdev_kpi] = calc_voltage_deviation_acopf(mpc_used, Pg2, ID, T, vref, vdev_fail_pen, vdev_fallback_to_pf, vdev_p_eps, vdev_pf_enforce_q);
    else
        [obj_vdev, vdev_kpi] = calc_voltage_deviation_acpf(mpc_used, Pg2, ID, T, vref, vdev_fail_pen, vdev_pf_enforce_q);
    end
end


% 存入 kpis
out.kpis.voltage_dev = vdev_kpi;   % 结构体：avg/max/failed_hours 等

%% 6.x) 气网安全性指标：节点压力 + 管道负载率
gas_kpi = struct();
gas_kpi.valid                 = false;   % 是否有有效气网结果
gas_kpi.press_violation_count = 0;
gas_kpi.press_violation_ratio = 0;
gas_kpi.press_violation_max_pu= 0;
gas_kpi.pipe_overload_count   = 0;
gas_kpi.pipe_overload_ratio   = 0;
gas_kpi.pipe_loading_max      = 0;

try
    if mpng_ok && exist('eg','var') && isstruct(eg) && isfield(eg,'mgc') && ~isempty(eg.mgc)
        mgc = eg.mgc;
        define_constants_gas;          % DEM, PR, PRMAX, PRMIN, FG_O, FMAX_O, ...
        tol = 1e-6;
        gas_kpi.valid = true;

        %% 1) 节点压力约束违背
        P    = [];
        Pmin = [];
        Pmax = [];

        if isfield(mgc,'node')
            if isstruct(mgc.node)
                % 参数表：优先 node.data，没有就退到 node.info
                Ninfo = [];
                if isfield(mgc.node,'info')
                    Ninfo = mgc.node.info;
                end
                if isfield(mgc.node,'data')
                    Ntab = mgc.node.data;
                else
                    Ntab = Ninfo;
                end
            else
                % 老版本：mgc.node 就是数值矩阵
                Ntab  = mgc.node;
                Ninfo = [];
            end

            if ~isempty(Ntab)
                % 实际压力：优先 info(PR)，否则用参数表里的 PR
                if ~isempty(Ninfo) && size(Ninfo,2) >= PR
                    P = Ninfo(:, PR);
                else
                    if size(Ntab,2) >= PR
                        P = Ntab(:, PR);
                    end
                end

                % 上下限：一般在参数表里
                if size(Ntab,2) >= PRMIN
                    Pmin = Ntab(:, PRMIN);
                end
                if size(Ntab,2) >= PRMAX
                    Pmax = Ntab(:, PRMAX);

                    % Optional "stress" knobs to intentionally create pressure violations (for testing/DRL)
                    if isfield(opts,'gas_pmin_scale') && ~isempty(opts.gas_pmin_scale) && exist('Pmin','var')
                        Pmin = Pmin * opts.gas_pmin_scale;
                    end
                    if isfield(opts,'gas_pmax_scale') && ~isempty(opts.gas_pmax_scale) && exist('Pmax','var')
                        Pmax = Pmax * opts.gas_pmax_scale;
                    end
                end
            end
        end

        % Optional: tighten KPI thresholds (does NOT change the solved constraints)
        if isfield(opts,'gas_kpi')
            if isfield(opts.gas_kpi,'pmax_scale') && ~isempty(opts.gas_kpi.pmax_scale)
                s = opts.gas_kpi.pmax_scale;
                if isfinite(s) && s > 0 && ~isempty(Pmax)
                    Pmax = Pmax .* s;
                end
            end
            if isfield(opts.gas_kpi,'pmin_scale') && ~isempty(opts.gas_kpi.pmin_scale)
                s = opts.gas_kpi.pmin_scale;
                if isfinite(s) && s > 0 && ~isempty(Pmin)
                    Pmin = Pmin .* s;
                end
            end
        end

        if ~isempty(P) && ~isempty(Pmin) && ~isempty(Pmax)
            n = min([numel(P), numel(Pmin), numel(Pmax)]);
            P    = P(1:n);
            Pmin = Pmin(1:n);
            Pmax = Pmax(1:n);

            viol_low  = P < (Pmin - 1e-3);
            viol_high = P > (Pmax + 1e-3);
            viol_any  = viol_low | viol_high;

            gas_kpi.press_violation_count = nnz(viol_any);
            gas_kpi.press_violation_ratio = gas_kpi.press_violation_count / n;

            under_pu = zeros(n,1);
            over_pu  = zeros(n,1);

            idxL = find(viol_low);
            idxH = find(viol_high);

            if ~isempty(idxL)
                under_pu(idxL) = (Pmin(idxL) - P(idxL)) ./ max(Pmin(idxL), tol);
            end
            if ~isempty(idxH)
                over_pu(idxH)  = (P(idxH) - Pmax(idxH)) ./ max(Pmax(idxH), tol);
            end

            gas_kpi.press_violation_max_pu = max([under_pu; over_pu; 0]);
        else
            % 没读到数据，用 NaN 表示“未知”
            gas_kpi.press_violation_count = 0;
            gas_kpi.press_violation_ratio = NaN;
            gas_kpi.press_violation_max_pu= NaN;
        end

        %% 2) 管道流量 / 负载率
        F    = [];
        Fmax = [];

        if isfield(mgc,'pipe') && ~isempty(mgc.pipe)
            Ptab = mgc.pipe;

            if isnumeric(Ptab)
                % 数值矩阵版本：直接按列号取
                if size(Ptab,2) >= FG_O
                    F = Ptab(:, FG_O);
                end
                if size(Ptab,2) >= FMAX_O
                    Fmax = Ptab(:, FMAX_O);
                end
            elseif isstruct(Ptab)
                % struct 版本：参数在 data，结果在 info
                if isfield(Ptab,'data') && size(Ptab.data,2) >= FMAX_O
                    Fmax = Ptab.data(:, FMAX_O);
                end
                if isfield(Ptab,'info') && size(Ptab.info,2) >= FG_O
                    F = Ptab.info(:, FG_O);
                end
            end
        end

        if isfield(opts,'gas_kpi')
            if isfield(opts.gas_kpi,'fmax_scale') && ~isempty(opts.gas_kpi.fmax_scale)
                s = opts.gas_kpi.fmax_scale;
                if isfinite(s) && s > 0 && ~isempty(Fmax)
                    Fmax = Fmax .* s;
                end
            end
        end

        if ~isempty(F) && ~isempty(Fmax)
            F     = F(:);
            Fmax  = Fmax(:);
            npipe = min(numel(F), numel(Fmax));
            F     = F(1:npipe);
            Fmax  = Fmax(1:npipe);

            % Optional stress knob: shrink pipe capacity to force overload (Fmax)
            if isfield(opts,'gas_fmax_scale') && ~isempty(opts.gas_fmax_scale)
                Fmax = Fmax * opts.gas_fmax_scale;
            end

            loading = abs(F) ./ max(Fmax, tol);   % 负载率 (p.u.)
            gas_kpi.pipe_loading_max    = max(loading);
            gas_kpi.pipe_overload_count = nnz(loading > 1 + 1e-3);
            gas_kpi.pipe_overload_ratio = gas_kpi.pipe_overload_count / npipe;
        else
            gas_kpi.pipe_loading_max    = NaN;
            gas_kpi.pipe_overload_count = 0;
            gas_kpi.pipe_overload_ratio = NaN;
        end
    else
        % mpng 未成功：标记结果无效
        gas_kpi.valid                 = false;
        gas_kpi.press_violation_count = 0;
        gas_kpi.press_violation_ratio = NaN;
        gas_kpi.press_violation_max_pu= NaN;
        gas_kpi.pipe_overload_count   = 0;
        gas_kpi.pipe_overload_ratio   = NaN;
        gas_kpi.pipe_loading_max      = NaN;
    end
catch ME
    warning('计算气网安全 KPI 时出错：%s', ME.message);
    gas_kpi.valid                 = false;
    gas_kpi.press_violation_ratio = NaN;
    gas_kpi.press_violation_max_pu= NaN;
    gas_kpi.pipe_overload_ratio   = NaN;
    gas_kpi.pipe_loading_max      = NaN;
end

% 6) 成本口径
E_load_MWh = E_gen_MWh;
avg_cost_per_MWh = NaN; avg_cost_per_h = NaN;

% 供电 / 未供电汇总
% 目前这套 MOST 模型没有显式“负荷丢失”变量，可以先近似认为：
%   - 所有发电量都被负荷消纳 ⇒ P_served = 总负荷电量
%   - 未供电量为 0
P_served    = E_load_MWh;
P_nonserved = 0;

if isfinite(cost_total) && ~isnan(E_load_MWh) && E_load_MWh>0
    avg_cost_per_MWh = cost_total / E_load_MWh;
    avg_cost_per_h   = cost_total / T;
end


% 7) 多目标
if isfinite(cost_total), obj_econ = -cost_total; end
if ~isnan(curtail_MWh), obj_curtail = -curtail_MWh; end
obj_smooth = 0;

% === 目标3：气网安全（最大化）===
% 说明：
%  - 有“越限/超载”时给很大的惩罚（硬惩罚）
%  - 没有越限时也给一点“靠近上限”的软惩罚（可选，用于reward shaping）
% gas_penalty = 0;
% if isfield(gas_kpi,'valid') && gas_kpi.valid == 1
%     % 硬惩罚权重（越限就重罚）
%     Wp_cnt   = 1e4;   % 压力越限节点数
%     Wp_amp   = 5e4;   % 压力越限幅度（pu）
%     Wf_cnt   = 1e4;   % 管道超载条数
%     Wf_amp   = 5e4;   % 超载幅度（loading-1）
%
%     % 软惩罚（可选：鼓励留裕度；不想要就把 Wmargin=0）
%     Wmargin  = 200;
%     margin_th = 0.85; % 例如希望 pipe_loading_max 尽量 < 0.85
%
%     pen_press = Wp_cnt*gas_kpi.press_violation_count + Wp_amp*gas_kpi.press_violation_max_pu;
%     pen_pipe  = Wf_cnt*gas_kpi.pipe_overload_count  + Wf_amp*max(0, gas_kpi.pipe_loading_max - 1);
%     pen_soft  = Wmargin*max(0, gas_kpi.pipe_loading_max - margin_th);
%
%     gas_penalty = pen_press + pen_pipe + pen_soft;
% else
%     % KPI 无效（例如求解失败/字段缺失）：直接重罚
%     gas_penalty = 1e6;
% end
%
% obj_gas_safe = -gas_penalty;
%
% % 把 penalty 存起来，方便你画图/调参/做消融
% gas_kpi.gas_penalty = gas_penalty;

%% === 目标3：气网安全（连续平滑版 reward shaping）===
% 目标：保留 gas_kpi 的统计输出（count/ratio/max 等），但 reward 计算不再依赖 nnz/max/硬阈值。
% 方法：
%   - 用 softplus(x) 平滑替代 max(0,x)
%   - 若存在向量级信息（pipe_loading_vec / press_under_pu_vec / press_over_pu_vec），
%     用 log-sum-exp 做“平滑 max”；否则退化用 *_max 标量。
%
% 注意：这不会让“OPF->reward”严格可导（优化器本身分段），但 reward 侧的非光滑点会大幅减少。

% ---------- 1) 超参数（可用 opts.reward 覆盖） ----------
beta = 40;            % 越大越接近硬阈值；20~60 常用。太大可能数值僵硬
margin_th = 0.85;     % 希望 pipe loading 尽量 < 0.85（留裕度）
fail_penalty = 1e6;   % 求解失败/字段缺失时的固定重罚（保持训练“远离不可行”）

% 权重（建议从小到大逐步调；先保证量纲别爆炸）
Wp_amp   = 5e4;   % 压力越限幅度（pu）的惩罚权重
Wf_over  = 5e4;   % 管道超载幅度（loading-1）的惩罚权重
Wmargin  = 2e2;   % 裕度惩罚（loading - margin_th）
Wratio_p = 5e3;   % 可选：压力越限比例（ratio）轻惩罚（注意 ratio 本身可能来自计数，权重不要大）
Wratio_f = 5e3;   % 可选：超载比例（ratio）轻惩罚

if isfield(opts,'reward') && isstruct(opts.reward)
    if isfield(opts.reward,'beta')        && isfinite(opts.reward.beta),        beta = opts.reward.beta; end
    if isfield(opts.reward,'margin_th')   && isfinite(opts.reward.margin_th),   margin_th = opts.reward.margin_th; end
    if isfield(opts.reward,'fail_penalty')&& isfinite(opts.reward.fail_penalty),fail_penalty = opts.reward.fail_penalty; end
    if isfield(opts.reward,'Wp_amp')      && isfinite(opts.reward.Wp_amp),      Wp_amp = opts.reward.Wp_amp; end
    if isfield(opts.reward,'Wf_over')     && isfinite(opts.reward.Wf_over),     Wf_over = opts.reward.Wf_over; end
    if isfield(opts.reward,'Wmargin')     && isfinite(opts.reward.Wmargin),     Wmargin = opts.reward.Wmargin; end
    if isfield(opts.reward,'Wratio_p')    && isfinite(opts.reward.Wratio_p),    Wratio_p = opts.reward.Wratio_p; end
    if isfield(opts.reward,'Wratio_f')    && isfinite(opts.reward.Wratio_f),    Wratio_f = opts.reward.Wratio_f; end
end

% ---------- 2) 平滑工具函数（数值稳定版 softplus / smoothmax） ----------
softplus = @(x) ( max(beta.*x, 0) + log1p(exp(-abs(beta.*x))) ) ./ beta;  % 近似 max(0,x)，处处可导
clip0    = @(x) max(0, double(x));                                        % 防 NaN/复数
safef    = @(x, d) (isfinite(x) && isreal(x)) .* double(x) + (~(isfinite(x)&&isreal(x))).*d;

smoothmax = @(v) local_smoothmax(v, beta);  % 见下方内联函数（末尾有定义）

% ---------- 3) 计算连续惩罚 ----------
gas_penalty = 0;

valid = isfield(gas_kpi,'valid') && (gas_kpi.valid == 1);
if valid
    % --- 压力越限幅度：优先使用向量（如果你在 KPI 段将它们存进 gas_kpi） ---
    p_amp = 0;  % 代表“最大越限幅度(pu)”的平滑版本
    if isfield(gas_kpi,'press_under_pu_vec') && ~isempty(gas_kpi.press_under_pu_vec)
        u = double(gas_kpi.press_under_pu_vec(:)); u = u(isfinite(u) & isreal(u));
        if ~isempty(u), p_amp = smoothmax(u); end
    end
    if isfield(gas_kpi,'press_over_pu_vec') && ~isempty(gas_kpi.press_over_pu_vec)
        o = double(gas_kpi.press_over_pu_vec(:));  o = o(isfinite(o) & isreal(o));
        if ~isempty(o), p_amp = max(p_amp, smoothmax(o)); end
    end
    if p_amp <= 0
        % 退化：用现有标量字段（你代码里已有 press_violation_max_pu）
        p_amp = safef(double(gas_kpi.press_violation_max_pu), 0);
    end
    p_amp = clip0(p_amp);

    % --- 管道负载：优先使用向量（如果你在 KPI 段存了 pipe_loading_vec） ---
    l_max = 0;
    if isfield(gas_kpi,'pipe_loading_vec') && ~isempty(gas_kpi.pipe_loading_vec)
        lv = double(gas_kpi.pipe_loading_vec(:)); lv = lv(isfinite(lv) & isreal(lv));
        if ~isempty(lv), l_max = smoothmax(lv); end
    end
    if l_max <= 0
        % 退化：用现有标量字段 pipe_loading_max
        l_max = safef(double(gas_kpi.pipe_loading_max), 0);
    end
    l_max = clip0(l_max);

    % --- 连续“越限”量（softplus） ---
    v_press  = softplus(p_amp);                % >=0，压力越限幅度
    v_over   = softplus(l_max - 1.0);          % >=0，超载幅度
    v_margin = softplus(l_max - margin_th);    % >=0，裕度不足（软惩罚）

    % --- 可选：ratio 项（注意 ratio 可能来自计数，不要给太大权重） ---
    r_p = 0; r_f = 0;
    if isfield(gas_kpi,'press_violation_ratio')
        r_p = safef(double(gas_kpi.press_violation_ratio), 0); r_p = clip0(r_p);
    end
    if isfield(gas_kpi,'pipe_overload_ratio')
        r_f = safef(double(gas_kpi.pipe_overload_ratio), 0);   r_f = clip0(r_f);
    end
    v_rp = softplus(r_p);   % 轻惩罚
    v_rf = softplus(r_f);   % 轻惩罚

    % --- 总惩罚：全部连续平滑（建议平方以增强梯度、抑制小噪声） ---
    gas_penalty = ...
        Wp_amp   * (v_press.^2) + ...
        Wf_over  * (v_over.^2)  + ...
        Wmargin  * (v_margin.^2) + ...
        Wratio_p * (v_rp.^2) + ...
        Wratio_f * (v_rf.^2);

else
    % KPI 无效（求解失败/字段缺失）：重罚（常数项不影响动作的“局部梯度”，但能强力引导策略避开不可行）
    gas_penalty = fail_penalty;
end

% 保险：防 NaN/复数
if ~isfinite(gas_penalty) || ~isreal(gas_penalty)
    gas_penalty = fail_penalty;
end

obj_gas_safe = -gas_penalty;     % 仍然是“最大化安全”（最小化惩罚）
gas_kpi.gas_penalty = gas_penalty;


% 8) 汇总输出结构
out.obj   = [obj_econ, obj_curtail, obj_gas_safe, obj_vdev];

% 兼容：w_obj 可能还是 3 维 ——> 自动补齐
w = w_obj(:);
if numel(w) < 4, w(end+1:4,1) = 0; end
out.score = w(:)' * out.obj(:);

out.cost_total     = cost_total;
out.curtail_MWh    = curtail_MWh;
out.curtail_MW     = curtail_MWh;
out.eval_for_drl   = @(cap) iter_couple_most_mpng_24h_merged(cap, w_obj).score;
out.comp_el_MW     = comp_el;
out.most_ok        = ok_most;

out.mpng_ok        = mpng_ok;
out.gas_ok         = mpng_ok;
out.ok             = logical(ok_most) && logical(mpng_ok);

% --- DRL-friendly KPI aliases (evaluate_cap expects these names) ---
avg_cost = avg_cost_per_MWh;
if ~isfinite(avg_cost), avg_cost = 1e6; end

curtail_ratio = 0;
if isfinite(ren_avail_MWh) && ren_avail_MWh > 0 && isfinite(curtail_MWh)
    curtail_ratio = curtail_MWh / ren_avail_MWh;
elseif isfinite(ren_util)
    curtail_ratio = max(0, 1 - ren_util);
end
curtail_ratio = min(max(curtail_ratio, 0), 1);

gas_risk = 1;   % invalid/failed gas run → worst
if isstruct(gas_kpi) && isfield(gas_kpi,'valid') && logical(gas_kpi.valid)
    vals = [];
    if isfield(gas_kpi,'press_violation_ratio') && isfinite(gas_kpi.press_violation_ratio)
        vals(end+1) = gas_kpi.press_violation_ratio;
    end
    if isfield(gas_kpi,'pipe_overload_ratio') && isfinite(gas_kpi.pipe_overload_ratio)
        vals(end+1) = gas_kpi.pipe_overload_ratio;
    end
    if isfield(gas_kpi,'press_violation_max_pu') && isfinite(gas_kpi.press_violation_max_pu)
        vals(end+1) = gas_kpi.press_violation_max_pu;
    end
    if isfield(gas_kpi,'pipe_loading_max') && isfinite(gas_kpi.pipe_loading_max)
        vals(end+1) = max(0, gas_kpi.pipe_loading_max - 1);
    end
    if isempty(vals)
        gas_risk = 0;
    else
        gas_risk = max(vals);
    end
end
gas_risk = min(max(gas_risk, 0), 1);

out.kpis = struct( ...
    'total_cost_USD',        cost_total, ...
    'avg_cost_per_MWh_USD',  avg_cost_per_MWh, ...
    'avg_cost',              avg_cost, ...
    'curtail_ratio',         curtail_ratio, ...
    'gas_risk',              gas_risk, ...
    'avg_cost_per_h_USD',    avg_cost_per_h, ...
    'load_energy_MWh',       E_load_MWh, ...
    'ren_avail_MWh',         ren_avail_MWh, ...
    'ren_sched_MWh',         ren_sched_MWh, ...
    'ren_utilization',       ren_util, ...
    'curtail_MWh',           curtail_MWh, ...
    'thermal_sched_MWh',     therm_sched_MWh, ...
    'profiles_applied',      profiles_ok, ...
    'gas',                   gas_kpi ...
    );

% --- [FIX] Preserve voltage deviation KPI computed above ---
if exist('vdev_kpi','var') && ~isempty(vdev_kpi) && isstruct(vdev_kpi)
    out.kpis.voltage_dev = vdev_kpi;
    if isfield(vdev_kpi,'avg'),          out.kpis.voltage_dev_avg_pu = vdev_kpi.avg; end
    if isfield(vdev_kpi,'max'),          out.kpis.voltage_dev_max_pu = vdev_kpi.max; end
    if isfield(vdev_kpi,'failed_hours'), out.kpis.voltage_dev_failed_hours = vdev_kpi.failed_hours; end
    if isfield(vdev_kpi,'src'),          out.kpis.voltage_dev_src = string(vdev_kpi.src); end
end
% 储能功率 & SOC & 绝对能量
out.Pg_storage_wind   = [];
out.Pg_storage_pv1    = [];
out.SOC_storage_wind  = [];
out.SOC_storage_pv1   = [];
out.En_storage_wind   = [];
out.En_storage_pv1    = [];

out.res_base_case   = eg;

% 功率（MW）
if exist('ts_ebat_wind','var') && ~isempty(ts_ebat_wind)
    out.Pg_storage_wind = ts_ebat_wind;
end
if exist('ts_ebat_pv1','var') && ~isempty(ts_ebat_pv1)
    out.Pg_storage_pv1  = ts_ebat_pv1;
end

% SOC（0~1）和能量（MWh）
if exist('SOC_storage_wind','var') && ~isempty(SOC_storage_wind)
    out.SOC_storage_wind = SOC_storage_wind;
end
if exist('SOC_storage_pv1','var') && ~isempty(SOC_storage_pv1)
    out.SOC_storage_pv1  = SOC_storage_pv1;
end
if exist('En_storage_wind','var') && ~isempty(En_storage_wind)
    out.En_storage_wind = En_storage_wind;
end
if exist('En_storage_pv1','var') && ~isempty(En_storage_pv1)
    out.En_storage_pv1 = En_storage_pv1;
end

setappdata(0,'boiler_report_txt',boiler_report_txt);
setappdata(0,'vre_overcap_msg',vre_msg);

%% ------------------- 打印（与原版口径一致） -------------------
try
    fprintf('=== 直观指标（24h） ===\n');
    if ~isnan(E_load_MWh)
        fprintf('负荷电量           = %.3f MWh\n', E_load_MWh);
    else
        fprintf('负荷电量           = NaN (Pg 未取到)\n');
    end
    if ~isnan(cost_total)
        fprintf('总成本             = %.3f $\n',   cost_total);
        if ~isnan(avg_cost_per_MWh), fprintf('单位电量成本       = %.3f $/MWh\n', avg_cost_per_MWh); end
        if ~isnan(avg_cost_per_h),   fprintf('平均时成本         = %.3f $/h\n',   avg_cost_per_h);   end
    end

    boiler_report_txt = getappdata(0,'boiler_report_txt');
    if ~isempty(boiler_report_txt), fprintf('%s\n', boiler_report_txt); end

    if ~isempty(Pg2)
        fprintf('可再生可用/调度    = %.3f / %.3f MWh，利用率=%.1f%%\n', ren_avail_MWh, ren_sched_MWh, 100*ren_util);
        vre_msg = getappdata(0,'vre_overcap_msg');
        if ~isempty(vre_msg), fprintf('%s\n', vre_msg); end
        fprintf('总弃风光           = %.3f MWh\n', curtail_MWh);
        fprintf('非可再生出力(合计) = %.3f MWh\n', therm_sched_MWh);
    else
        fprintf('总弃风光(退化)     = %.3f MWh\n', curtail_MWh);
    end

    if ~isempty(Pg2)
        try
            [~, ordX] = ext2int(mpc_used); e2i = ordX.gen.e2i;
        catch
            e2i = (1:size(Pg2,1))';
        end
        pickE = @(ext_row) sum(Pg2(e2i(ext_row),:), 'all');

        E_wind  = pickE(idx_wind);
        E_pv1   = pickE(idx_pv1);
        E_pv2   = pickE(idx_pv2);
        E_gas   = pickE(idx_gas);
        E_slack = pickE(idx_slack);

        all_ext   = (1:size(mpc.gen,1))';
        known_ext = [idx_wind; idx_pv1; idx_pv2; idx_gas; idx_slack];
        other_ext = setdiff(all_ext, known_ext);
        E_other = 0; name_other = {};
        for ii = 1:numel(other_ext)
            r = other_ext(ii);
            try
                E_other = E_other + pickE(r);
                name_other{end+1} = sprintf('GEN[%d]@bus%d', r, mpc.gen(r, ID.GEN_BUS));
            catch
            end
        end

        fprintf('--- 各机组能量 ---\n');
        fprintf('WIND  = %.3f MWh\n', E_wind);
        fprintf('PV1   = %.3f MWh\n', E_pv1);
        fprintf('PV2   = %.3f MWh\n', E_pv2);
        fprintf('GAS   = %.3f MWh\n', E_gas);
        fprintf('SLACK = %.3f MWh\n', E_slack);
        if E_other > 1e-3
            fprintf('OTHER = %.3f MWh  （%s）\n', E_other, strjoin(name_other, ', '));
        end

        E_sum = E_wind + E_pv1 + E_pv2 + E_gas + E_slack + E_other;
        if abs(E_sum - E_gen_MWh) > 1e-3
            fprintf('【提示】分机组能量求和 %.3f 与总能量 %.3f 不一致，请检查 ext↔int 映射。\n', E_sum, E_gen_MWh);
        end
    end

    if ~profiles_ok
        fprintf('【提示】MOST 未成功应用 profiles（PMAX/负荷时序），已按基准/裁剪口径统计。\n');
    end

    fprintf('--- 多目标指标 [econ, curtail, gas_safe, obj_vdev] ---\n');
    fprintf('%10.4f %10.4f %10.4f %10.4f\n', out.obj(1), out.obj(2), out.obj(3), out.obj(4));
    fprintf('聚合得分 score = %.6f\n', out.score);
catch
end

out.vis.Pg2       = Pg2;
out.vis.mpc_used  = mpc_used;
out.vis.idx_wind  = idx_wind;
out.vis.idx_pv1   = idx_pv1;
out.vis.idx_pv2   = idx_pv2;
out.vis.idx_gas   = idx_gas;
out.vis.idx_slack = idx_slack;
out.vis.el_boiler = el_boiler;
out.vis.pmx_wind  = pmx_wind;
out.vis.pmx_pv1   = pmx_pv1;
out.vis.pmx_pv2   = pmx_pv2;

% 供图用的附加信息：GRID 索引、总负荷时序、分时电价系数
out.vis.idx_grid = idx_grid;                                 % GRID 机组行号（当前是 1）
out.vis.P_load   = Pd_base * scale(:) + el_boiler(:);        % 近似总负荷 (MW)
out.vis.lambda_e = lambda_e(:);                              % 分时电价标幺

out.vis.E_wind    = E_wind;
out.vis.E_pv1     = E_pv1;
out.vis.E_pv2     = E_pv2;
out.vis.E_gas     = E_gas;
out.vis.E_slack   = E_slack;
out.vis.E_other   = E_other;
out.vis.P_served  = P_served;
out.vis.P_nonserved = P_nonserved;


end % ===== 主函数结束 =====


%% ================== 工具函数 ==================
function mdE = try_loadmd_multi(mpc, T, profiles, xgd, sd)
% 封装 loadmd 调用，自动兼容“有/无 profile、有/无 xgd/sd”等多种组合
if nargin < 4 || isempty(xgd), xgd = []; end
if nargin < 5 || isempty(sd),  sd  = []; end

lastErr = '';

% 1) 优先尝试 (mpc, T, xgd, sd, profiles)
try
    mdE = loadmd(mpc, T, xgd, sd, profiles);
    return;
catch ME1
    lastErr = ME1.message;
end

% 2) 再尝试 6 参版本 (mpc, T, xgd, sd, [], profiles)
try
    mdE = loadmd(mpc, T, xgd, sd, [], profiles);
    return;
catch ME2
    lastErr = catmsg(lastErr, ME2.message);
end

% 3) 最后兜底：不用 xgd/sd，只为了兼容非常老的 loadmd
try
    mdE = loadmd(mpc, T, struct(), struct(), profiles);
    return;
catch ME3
    lastErr = catmsg(lastErr, ME3.message);
end

error(lastErr);
end


function warn_slim(fmt, varargin)
% 精简版 warning（避免重复打印的冗长堆栈）
msg = sprintf(fmt, varargin{:});
msg = regexprep(msg, '\s+', ' ');
warning('%s', msg);
end

function ID = get_indices()
[~, ~, PD, QD, GS, BS, BUS_AREA, VM] = idx_bus;
[GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, ...
    ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, RAMP_AGC, RAMP_10, RAMP_30] = idx_gen;
[~, ~, RATE_A] = idx_brch;
ID.PD=PD; ID.QD=QD; ID.GEN_BUS=GEN_BUS; ID.PG=PG; ID.QG=QG;
ID.QMAX=QMAX; ID.QMIN=QMIN; ID.VG=VG; ID.MBASE=MBASE;
ID.GEN_STATUS=GEN_STATUS; ID.PMAX=PMAX; ID.PMIN=PMIN;
ID.GS=GS; ID.BS=BS; ID.VM=VM; ID.RATE_A=RATE_A;
ID.RAMP_AGC=RAMP_AGC; ID.RAMP_10=RAMP_10; ID.RAMP_30=RAMP_30;
ID.BUS_AREA = BUS_AREA;
end

function [mpc, idx] = safe_add_gen(mpc, bus, Pmax, Qmax, Vref, kind, ID)
ncols = size(mpc.gen,2);
g = zeros(1,ncols);
g(ID.GEN_BUS)=bus; g(ID.PG)=0; g(ID.QG)=0; g(ID.QMAX)=Qmax; g(ID.QMIN)=-Qmax;
g(ID.VG)=Vref; g(ID.MBASE)=mpc.baseMVA; g(ID.GEN_STATUS)=1; g(ID.PMAX)=Pmax; g(ID.PMIN)=0;
mpc.gen = [mpc.gen; g]; idx = size(mpc.gen,1);
row7 = [2 0 0 3 0 1e-5 0];
if isempty(mpc.gencost)
    mpc.gencost = row7;
else
    nc = size(mpc.gencost,2);
    if nc < 7, mpc.gencost = [mpc.gencost zeros(size(mpc.gencost,1), 7-nc)]; nc = 7; end
    r = zeros(1,nc); r(1:7) = row7(1:7);
    mpc.gencost = [mpc.gencost; r];
end
if isfield(mpc,'gentype'), mpc.gentype{end+1,1} = upper(kind); end
if isfield(mpc,'genfuel')
    switch lower(kind)
        case 'pv',   mpc.genfuel{end+1,1} = 'solar';
        case 'wind', mpc.genfuel{end+1,1} = 'wind';
        otherwise,   mpc.genfuel{end+1,1} = 'other';
    end
end
end

function row = pad_gencost_linear(gencost, c1)
nc = size(gencost,2);
if nc < 7, gencost = [gencost zeros(size(gencost,1), 7-nc)]; nc = 7; end
row = zeros(1, max(7,nc)); row(1:7) = [2 0 0 3 0 c1 0];
end

function gencost = enforce_linear_cost(gencost, ngen, c1)
if isempty(gencost), gencost = zeros(0,7); end
if size(gencost,2) < 7, gencost(:,7) = 0; end
if size(gencost,1) < ngen
    need = ngen - size(gencost,1);
    gencost = [gencost; repmat(gencost(1,:), need, 1)];
end
for gi = 1:ngen
    if all(gencost(gi,5:7)==0), gencost(gi,1:7) = [2 0 0 3 0 c1 0]; end
end
end

function s = shape_pv(T)
t = linspace(0,1,T);
s = max(0,sin(pi*t)).^2;
s = s / max(1e-6, max(s));
end

function s = shape_wind(T)
t = linspace(0,1,T);
s = 0.6 + 0.4*sin(2*pi*(t-0.2));
s = max(0,s);
s = s / max(1e-6, max(s));
end

% 建议直接把 shape_heat.m 改成如下（T=24 时）：
function y = shape_heat(T)
base24 = [ ...
    0.80  % 1
    0.75  % 2
    0.70  % 3
    0.65  % 4   % 夜谷更低
    0.70  % 5
    0.80  % 6
    0.90  % 7
    1.00  % 8
    1.08  % 9
    1.15  % 10
    1.18  % 11
    1.20  % 12
    1.15  % 13
    1.05  % 14
    0.98  % 15
    0.95  % 16
    1.05  % 17
    1.18  % 18
    1.25  % 19
    1.20  % 20
    1.00  % 21  % 晚峰后直接掉
    0.90  % 22
    0.80  % 23
    0.70];% 24 % 接近夜谷，但不完全水平

if T == 24
    y = base24(:);
else
    % 若 T≠24，做一个线性插值
    t24 = (1:24)';
    t   = linspace(1,24,T)';
    y   = interp1(t24, base24(:), t, 'pchip');
end
end

function Pg = take_Pg(res)
Pg = [];
try
    if ~isstruct(res) || ~isfield(res, 'results'), return; end
    R = res.results;

    NG = []; Tt = [];
    try, R1 = R.GenTLMP; [NGg, Tg] = size(R1); NG = NGg; Tt = Tg; catch, end

    cand = try_get(R,'ExpectedDispatch','Pg');
    if isempty(cand), cand = try_get(R,'Dispatch','Pg'); end
    if isempty(cand), cand = try_get(R,'Expected','Dispatch','Pg'); end
    if isempty(cand), cand = try_get(R,'Pg'); end
    if ~isempty(cand)
        Pg = orient_clip(cand, NG, Tt); if ~isempty(Pg), fprintf('[take_Pg] 来源：ExpectedDispatch 数值矩阵\n'); return; end
    end

    if isfield(R,'ExpectedDispatch') && isnumeric(R.ExpectedDispatch)
        E = double(R.ExpectedDispatch);
        Pg_try = orient_clip(E, NG, Tt);
        if ~isempty(Pg_try), Pg = Pg_try; fprintf('[take_Pg] 来源：ExpectedDispatch 数值矩阵\n'); return; end
    end
catch
end
end

function v = try_get(S, varargin)
v = [];
for k = 1:numel(varargin)
    f = varargin{k};
    if isstruct(S) && isfield(S,f), S = S.(f); else, v = []; return; end
end
if isnumeric(S), v = double(S); end
end

function X = orient_clip(V, NG, T)
X = [];
if ~isnumeric(V) || isempty(V), return; end
if ~isempty(NG) && ~isempty(T)
    if isequal(size(V), [NG, T]), X = V; return; end
    if isequal(size(V), [T, NG]), X = V.'; return; end
    if isvector(V) && numel(V) == NG*T, X = reshape(V, [NG, T]); return; end
else
    [r,c] = size(V);
    if r <= c && r <= 128, X = V; return; end
    if c <  r && c <= 128, X = V.'; return; end
end
end

function A = safe_pick_rows(Pg, rows_int, rows_ext)
A = zeros(numel(rows_ext), size(Pg,2));
rows_int = rows_int(:);
for i = 1:numel(rows_ext)
    if i <= numel(rows_int) && rows_int(i)>=1 && rows_int(i)<=size(Pg,1)
        A(i,:) = Pg(rows_int(i),:);
    else
        A(i,:) = 0;
    end
end
end

function s = catmsg(a, b)
if isempty(a), s = b; else, s = [a, '；', b]; end
end

%% ---- 统一为 NB×T + 提供 T×NB 别名（电侧） ----
function mpgc = normalize_mpng_case(mpgc, mpc_used, ID, T)
if isfield(mpgc,'mpc') && isstruct(mpgc.mpc)
    msrc = mpgc.mpc;
else
    msrc = mpc_used;
    mpgc.mpc = msrc;
end
core = {'bus','gen','branch','gencost','areas','baseMVA'};
for k = 1:numel(core)
    f = core{k}; if isfield(msrc, f), mpgc.(f) = msrc.(f); end
end

NB = size(mpgc.bus, 1);
PD0 = mpgc.bus(:, ID.PD);
QD0 = mpgc.bus(:, ID.QD);

pd_NT = repmat(PD0, 1, T);
qd_NT = repmat(QD0, 1, T);
pd_TN = pd_NT.'; qd_TN = qd_NT.';

if ~isfield(mpgc,'connect') || ~isstruct(mpgc.connect), mpgc.connect = struct(); end
if ~isfield(mpgc.connect,'power') || ~isstruct(mpgc.connect.power), mpgc.connect.power = struct(); end

DemP = struct();
DemP.pd    = pd_NT;    DemP.qd    = qd_NT;
DemP.PDt   = pd_NT;    DemP.QDt   = qd_NT;
DemP.pd_TN = pd_TN;    DemP.qd_TN = qd_TN;
mpgc.connect.power.demands = DemP;
mpgc.connect.power.buses   = (1:NB).';
mpgc.connect.power.nbuses  = NB;

if ~isfield(mpgc.connect.power,'time') || isempty(mpgc.connect.power.time)
    mpgc.connect.power.time = ones(1, T);
else
    mpgc.connect.power.time = reshape(mpgc.connect.power.time, 1, []);
end
mpgc.connect.power.dt        = 1;
mpgc.connect.power.day_hours = sum(mpgc.connect.power.time);
mpgc.connect.power.NT        = T;

mpgc.connect.time      = mpgc.connect.power.time;
mpgc.connect.dt        = 1;
mpgc.connect.day_hours = sum(mpgc.connect.power.time);
mpgc.connect.NT        = T;

% --- normalize field names for MPNG (matpower/matgas/connect) ---
% matgas: accept mgc/matgas_case/gas aliases
if (~isfield(mpgc,'matgas') || isempty(mpgc.matgas))
    if isfield(mpgc,'mgc') && ~isempty(mpgc.mgc)
        mpgc.matgas = mpgc.mgc;
    elseif isfield(mpgc,'matgas_case') && ~isempty(mpgc.matgas_case)
        mpgc.matgas = mpgc.matgas_case;
    elseif isfield(mpgc,'gas') && ~isempty(mpgc.gas)
        mpgc.matgas = mpgc.gas;
    end
end
% keep mgc as an alias too (some downstream code may still use it)
if (~isfield(mpgc,'mgc') || isempty(mpgc.mgc))
    if isfield(mpgc,'matgas') && ~isempty(mpgc.matgas)
        mpgc.mgc = mpgc.matgas;
    end
end
% matpower/mpc aliasing
if ~isfield(mpgc,'matpower') && isfield(mpgc,'mpc') && ~isempty(mpgc.mpc)
    mpgc.matpower = mpgc.mpc;
elseif ~isfield(mpgc,'mpc') && isfield(mpgc,'matpower') && ~isempty(mpgc.matpower)
    mpgc.mpc = mpgc.matpower;
end

fprintf('[MPNG-NORM] NB=%d, pd(NB×T)=%dx%d\n', NB, size(DemP.pd,1), size(DemP.pd,2));
end

%% ---- 稳健识别 NB_g，并提供 NB_g×T 与 T×NB_g（气侧） ----
function mpgc = attach_gas_demands(mpgc, mgc_case, T)
NB_g = detect_gas_nodes(mgc_case);
if NB_g <= 0
    error('attach_gas_demands: 无法确定气网母线数。');
end

pdg_NT = zeros(NB_g, T);
pdg_TN = zeros(T, NB_g);

if ~isfield(mpgc,'connect') || ~isstruct(mpgc.connect), mpgc.connect = struct(); end
mpgc.connect.gas = struct();
mpgc.connect.gas.demands = struct();
mpgc.connect.gas.demands.pd     = pdg_NT;
mpgc.connect.gas.demands.PDt    = pdg_NT;
mpgc.connect.gas.demands.pd_TN  = pdg_TN;

mpgc.connect.gas.buses   = (1:NB_g).';
mpgc.connect.gas.nbuses  = NB_g;
mpgc.connect.gas.time    = ones(1, T);
mpgc.connect.gas.dt      = 1;
mpgc.connect.gas.day_hours = sum(mpgc.connect.gas.time);
mpgc.connect.gas.NT      = T;

fprintf('[MPNG-NORM-GAS] NB_g=%d, gas pd(NB×T)=%dx%d\n', NB_g, size(pdg_NT,1), size(pdg_NT,2));
end


function mpgc = enforce_mpng_top_fields(mpgc, mpc_used, mgc_case, connect)
% 强制补齐 MPNG 入口要求的顶层字段：connect + matgas（并保持 matpower/mpc 一致）
if nargin < 4, connect = []; end
if nargin < 3, mgc_case = []; end
if nargin < 2, mpc_used = []; end

% connect
if (~isfield(mpgc,'connect') || isempty(mpgc.connect)) && ~isempty(connect)
    mpgc.connect = connect;
end

% matpower/mpc
if (~isfield(mpgc,'matpower') || isempty(mpgc.matpower)) && ~isempty(mpc_used)
    mpgc.matpower = mpc_used;
end
if (~isfield(mpgc,'mpc') || isempty(mpgc.mpc))
    if isfield(mpgc,'matpower') && ~isempty(mpgc.matpower)
        mpgc.mpc = mpgc.matpower;
    elseif ~isempty(mpc_used)
        mpgc.mpc = mpc_used;
    end
end

% matgas/mgc
if (~isfield(mpgc,'matgas') || isempty(mpgc.matgas))
    if isfield(mpgc,'mgc') && ~isempty(mpgc.mgc)
        mpgc.matgas = mpgc.mgc;
    elseif ~isempty(mgc_case)
        mpgc.matgas = mgc_case;
    end
end
if (~isfield(mpgc,'mgc') || isempty(mpgc.mgc))
    if isfield(mpgc,'matgas') && ~isempty(mpgc.matgas)
        mpgc.mgc = mpgc.matgas;
    elseif ~isempty(mgc_case)
        mpgc.mgc = mgc_case;
    end
end
end

function NB_g = detect_gas_nodes(mg)
NB_g = 0;
if isfield(mg,'node') && isstruct(mg.node) && isfield(mg.node,'info') && ~isempty(mg.node.info)
    NB_g = size(mg.node.info,1); if NB_g>0, return; end
end
names_first = {'node','nodes','junction','junctions','bus','buses'};
for i=1:numel(names_first)
    f = names_first{i};
    if isfield(mg,f) && ~isempty(mg.(f))
        NB_g = size(mg.(f),1); if NB_g>0, return; end
    end
end
names_2 = {'branch','pipe','pipes','line','lines','comp','compressor','compressors','well','wells','sto','storage','loads','demand','demands'};
for i=1:numel(names_2)
    f = names_2{i};
    if isfield(mg,f) && ~isempty(mg.(f))
        A = mg.(f);
        if size(A,2)>=2
            NB_g = max(NB_g, max(max(A(:,1)), max(A(:,2))));
        else
            NB_g = max(NB_g, max(A(:)));
        end
    end
end
end

function S = ensure_mpng_shapes(S)
if isfield(S,'connect') && isfield(S.connect,'power') && isfield(S.connect.power,'demands')
    D = S.connect.power.demands;
    if isfield(D,'pd'),  D.pd  = full(double(D.pd));  D.pd(~isfinite(D.pd))=0; end
    if isfield(D,'qd'),  D.qd  = full(double(D.qd));  D.qd(~isfinite(D.qd))=0; end
    if isfield(D,'PDt'), D.PDt = full(double(D.PDt)); D.PDt(~isfinite(D.PDt))=0; end
    if isfield(D,'QDt'), D.QDt = full(double(D.QDt)); D.QDt(~isfinite(D.QDt))=0; end
    S.connect.power.demands = D;
end
if isfield(S,'connect') && isfield(S.connect,'gas') && isfield(S.connect.gas,'demands')
    G = S.connect.gas.demands;
    if isfield(G,'pd'),  G.pd  = full(double(G.pd));  G.pd(~isfinite(G.pd))=0; end
    if isfield(G,'PDt'), G.PDt = full(double(G.PDt)); G.PDt(~isfinite(G.PDt))=0; end
    S.connect.gas.demands = G;
end
end

function tf = ok_power_demands(S)
tf = false;
if ~(isfield(S,'bus') && isfield(S,'connect') && isfield(S.connect,'power') && isfield(S.connect.power,'demands')), return; end
NB = size(S.bus,1);
D  = S.connect.power.demands;
if isfield(D,'pd') && size(D.pd,1)==NB, tf = true; return; end
if isfield(D,'PDt') && size(D.PDt,1)==NB, tf = true; return; end
end

function cands = build_mpng_payloads(mpc_used, mgc_case, connect, ID, T)
cands = {};

% A) 官方签名（单结构体，由 mpc2gas_prep 返回）
try
    mpgc = mpc2gas_prep(mpc_used, mgc_case, connect);
    mpgc = normalize_mpng_case(mpgc, mpc_used, ID, T);
    mpgc = attach_gas_demands(mpgc, mgc_case, T);
    mpgc = ensure_mpng_shapes(mpgc);
    cands{end+1} = mpgc;
catch
end

% B) 单结构体签名（我们手工组装）
S = struct('mpc', mpc_used, 'mgc', mgc_case, 'connect', connect);
try
    mpgc = mpc2gas_prep(S);
    mpgc = normalize_mpng_case(mpgc, mpc_used, ID, T);
    mpgc = attach_gas_demands(mpgc, mgc_case, T);
    mpgc = ensure_mpng_shapes(mpgc);
    cands{end+1} = mpgc;
catch
    try
        mpgc = normalize_mpng_case(S, mpc_used, ID, T);
        mpgc = attach_gas_demands(mpgc, mgc_case, T);
        mpgc = ensure_mpng_shapes(mpgc);
        cands{end+1} = mpgc;
    catch
    end
end

% C) 顶层扁平（依然单结构体）
mpgc2 = struct();
mpgc2.mpc      = mpc_used;
mpgc2.mgc      = mgc_case;
mpgc2.matgas   = mgc_case;
mpgc2.matpower = mpc_used;
mpgc2.connect  = connect;
core = {'bus','gen','branch','gencost','areas','baseMVA'};
for k = 1:numel(core)
    f = core{k}; if isfield(mpc_used, f), mpgc2.(f) = mpc_used.(f); end
end
mpgc2 = normalize_mpng_case(mpgc2, mpc_used, ID, T);
mpgc2 = attach_gas_demands(mpgc2, mgc_case, T);
mpgc2 = ensure_mpng_shapes(mpgc2);
cands{end+1} = mpgc2;

% D) 极简直传（依旧作为单结构体）
mpgc3 = struct();
mpgc3.connect = mpgc2.connect;
mpgc3.matgas  = mgc_case;
mpgc3.mpc     = mpc_used;
mpgc3.bus     = mpc_used.bus;
mpgc3.gen     = mpc_used.gen;
mpgc3.branch  = mpc_used.branch;
mpgc3.baseMVA = mpc_used.baseMVA;
mpgc3 = normalize_mpng_case(mpgc3, mpc_used, ID, T);
mpgc3 = attach_gas_demands(mpgc3, mgc_case, T);
mpgc3 = ensure_mpng_shapes(mpgc3);
cands{end+1} = mpgc3;

keep = true(1, numel(cands));
for i = 1:numel(cands)
    S2 = cands{i};
    % 只保留满足 MPNG 入口检查的候选：必须包含 connect + matgas
    has_matgas  = isfield(S2,'matgas') && ~isempty(S2.matgas);
    has_connect = isfield(S2,'connect') && ~isempty(S2.connect);
    keep(i) = has_matgas && has_connect && ok_power_demands(S2) && isfield(S2,'bus') && isfield(S2,'gen') && isfield(S2,'branch');
end
cands = cands(keep);
end

function v = safe_num(S, path, def)
% 读取 S.path（点号路径），失败返回 def
try
    parts = strsplit(path,'.');
    v = S;
    for i=1:numel(parts)
        f = parts{i};
        if isstruct(v) && isfield(v,f), v = v.(f); else, v = def; return; end
    end
    if ~isnumeric(v) || isempty(v), v = def; end
catch
    v = def;
end
end

%% -------- 气网清单打印 --------
function print_gas_inventory(mgc)
define_constants_gas;
fprintf('\n===== 48 节点气网清单 =====\n');

% ---- 节点表 ----
if isfield(mgc,'node') && isfield(mgc.node,'info') && ~isempty(mgc.node.info)
    N = mgc.node.info;
    fprintf('[NODES] 共 %d 个节点：\n', size(N,1));
    for i = 1:size(N,1)
        row = N(i,:);
        head = sprintf(' node[%02d]:', i);
        fprintf('%-10s ', head);
        fprintf('%g ', row);
        fprintf('\n');
    end
else
    fprintf('[NODES] 未检测到 mgc.node.info（跳过）。\n');
end

% ---- 井/气源 ----
if isfield(mgc,'well') && ~isempty(mgc.well)
    W = mgc.well;
    fprintf('[WELLS] 共 %d 口气井：\n', size(W,1));
    for k = 1:size(W,1)
        row = W(k,:);
        from_node = safe_col(row,1);
        fprintf('  well[%02d] @node=%g | ', k, from_node);
        fprintf('%g ', row);
        fprintf('\n');
    end
else
    fprintf('[WELLS] 无记录。\n');
end

% ---- 管道 ----
if isfield(mgc,'pipe') && ~isempty(mgc.pipe)
    P = mgc.pipe;
    fprintf('[PIPES] 共 %d 条管道（显示 from->to 及前4列信息）：\n', size(P,1));
    for k = 1:size(P,1)
        row = P(k,:);
        f = safe_col(row,1); t = safe_col(row,2);
        fprintf('  pipe[%02d] %g -> %g | ', k, f, t);
        fprintf('%g ', row(1:min(4,end)));
        if numel(row) > 4, fprintf(' ...'); end
        fprintf('\n');
    end
else
    fprintf('[PIPES] 无记录。\n');
end

% ---- 压缩机 ----
if isfield(mgc,'comp') && ~isempty(mgc.comp)
    C = mgc.comp;
    fprintf('[COMPS] 共 %d 台压缩机：\n', size(C,1));
    for k = 1:size(C,1)
        row = C(k,:);
        typ = NaN;
        try, typ = row(TYPE_C); catch, end
        label = '(unknown)';
        if ~isnan(typ)
            if     typ==COMP_P, label='COMP_P(电驱)';
            elseif typ==COMP_G, label='COMP_G(气驱)';
            end
        end
        f = safe_col(row,1); t = safe_col(row,2);
        ratio = NaN; pc = NaN;
        try, ratio = row(RATIO_C); catch, end
        try, pc    = row(PC_C);    catch, end
        fprintf('  comp[%02d] %g -> %g | type=%s', k, f, t, label);
        if ~isnan(ratio), fprintf(', ratio=%.3f', ratio); end
        if ~isnan(pc),    fprintf(', Pc(MW)=%.3f', pc); end
        fprintf('\n');
    end
else
    fprintf('[COMPS] 无记录。\n');
end

% ---- 储气 ----
if isfield(mgc,'sto') && ~isempty(mgc.sto)
    S = mgc.sto;
    fprintf('[STO] 共 %d 处储气：\n', size(S,1));
    for k = 1:size(S,1)
        row = S(k,:);
        at = safe_col(row,1);
        fprintf('  sto[%02d] @node=%g | ', k, at);
        fprintf('%g ', row);
        fprintf('\n');
    end
else
    fprintf('[STO] 无记录。\n');
end

fprintf('===== 气网清单结束 =====\n\n');
end

function v = safe_col(row, idx)
if idx <= numel(row), v = row(idx); else, v = NaN; end
end


function Y = take_ts(Pg2, mpc, rows_ext)
% 取若干外部行的时间序列并堆叠（k×T）
k = numel(rows_ext);
Y = zeros(k, size(Pg2,2));
try
    [~, ord] = ext2int(mpc); e2i = ord.gen.e2i;
    for i=1:k
        r = rows_ext(i);
        if r>=1 && r<=size(mpc.gen,1)
            Y(i,:) = Pg2(e2i(r),:);
        end
    end
catch
    for i=1:k
        r = rows_ext(i);
        if r>=1 && r<=size(Pg2,1)
            Y(i,:) = Pg2(r,:);
        end
    end
end
end

function E = local_reconstruct_E(P, E0, Emax, eta_ch, eta_dis, dt, T)
% 根据功率时序 P (MW) + 效率 + 步长 dt 重建能量轨迹 E (MWh)
% P>0 放电，P<0 充电
P  = double(P(:).');    % 行向量
E  = zeros(1, T);
E(1) = E0;

for tt = 1:T-1
    p   = P(tt);
    ch  = max(-p, 0);   % 充电功率 (MW)
    dis = max( p, 0);   % 放电功率 (MW)

    dE = (eta_ch * ch - dis / max(eta_dis, eps)) * dt;  % MWh
    E(tt+1) = E(tt) + dE;

    % 夹在 [0, Emax] 范围内，防止数值跑飞
    if Emax > 0
        E(tt+1) = min(Emax, max(0, E(tt+1)));
    end
end
end

% ================= Gas safety scoring helper =================
function gas_risk = gas_risk_score(gas_kpi)
% Returns a NON-negative scalar. Larger = worse gas network security.
% Designed to be finite even if gas_kpi contains NaN.
gas_risk = 0;

try
    valid = isfield(gas_kpi,'valid') && (gas_kpi.valid == 1);
    if ~valid
        gas_risk = gas_risk + 1e3;
    end

    pr_ratio = safe_num(getfield_def(gas_kpi,'press_violation_ratio',0), 0); %#ok<GFLD>
    pr_maxpu = safe_num(getfield_def(gas_kpi,'press_violation_max_pu',0), 0); %#ok<GFLD>
    po_ratio = safe_num(getfield_def(gas_kpi,'pipe_overload_ratio',0), 0); %#ok<GFLD>
    pl_max   = safe_num(getfield_def(gas_kpi,'pipe_loading_max',0), 0); %#ok<GFLD>

    gas_risk = gas_risk + 200*pr_ratio + 200*max(pr_maxpu,0);
    gas_risk = gas_risk + 100*po_ratio + 200*max(pl_max-1,0);
catch
    gas_risk = 1e6; % absolute fail-safe
end

if ~isfinite(gas_risk) || ~isreal(gas_risk)
    gas_risk = 1e6;
end
end

function v = getfield_def(s, f, d)
if isstruct(s) && isfield(s,f)
    v = s.(f);
else
    v = d;
end
end


% ---------- 内联：平滑 max（log-sum-exp） ----------
function m = local_smoothmax(v, beta)
v = double(v(:));
v = v(isfinite(v) & isreal(v));
if isempty(v)
    m = 0;
    return;
end
% log-sum-exp 稳定实现：max(v) + (1/beta)*log(sum(exp(beta*(v-max(v)))))
vmax = max(v);
m = vmax + (1/beta) * log(sum(exp(beta*(v - vmax))));
if ~isfinite(m) || ~isreal(m), m = vmax; end
end

function [obj_vdev, kpi] = calc_voltage_deviation_from_mpng(eg, ID, T, vref, fail_pen, nb_base)
% 直接从 MPNG 的 AC-OPF 结果中提取 Vm 并计算电压偏差（优先路径）
% 兼容两类返回：
%   1) eg.mpc.bus 为 “nb_base*T” 行（多时段扩展堆叠）
%   2) eg.mpc.bus 仅 “nb_base” 行（单时段快照/最后一时段）
% 若无法识别形状，则标记 used=false，交由上层回退到 acopf/acpf。

if nargin < 5 || isempty(fail_pen), fail_pen = 0.2; end
if nargin < 4 || isempty(vref),     vref     = 1.0; end
if nargin < 3 || isempty(T),        T        = 24;  end
if nargin < 6 || isempty(nb_base),  nb_base  = [];  end

kpi = struct();
kpi.used         = false;
kpi.per_hour     = ones(T,1) * fail_pen;
kpi.failed_hours = T;
kpi.avg          = fail_pen;
kpi.max          = fail_pen;
kpi.src          = "mpng:init";

if isempty(eg) || ~isstruct(eg) || ~isfield(eg,'mpc') || ~isstruct(eg.mpc) || ~isfield(eg.mpc,'bus')
    kpi.src = "mpng:no_mpc";
    obj_vdev = kpi.avg; return;
end

bus = eg.mpc.bus;
if isempty(bus) || size(bus,2) < ID.VM
    kpi.src = "mpng:no_vm_col";
    obj_vdev = kpi.avg; return;
end

Vm_all = bus(:, ID.VM);
Vm_all = Vm_all(:);
nb_total = size(bus,1);

if isempty(nb_base)
    nb_base = nb_total;  % 兜底：按单时段处理
end

H = []; nb_per = [];
mode = "";

% (A) 标准：nb_total == nb_base*T
if nb_total == nb_base * T
    H = T; nb_per = nb_base; mode = "stack_nbxt";
    % (B) 单时段快照：nb_total == nb_base
elseif nb_total == nb_base
    H = 1; nb_per = nb_base; mode = "single_snapshot";
    % (C) 可推断：nb_total 可被 nb_base 整除（但时段数不等于 T）
elseif mod(nb_total, nb_base) == 0
    H = nb_total / nb_base; nb_per = nb_base; mode = "infer_H_from_nb";
    % (D) 另一种可推断：nb_total 可被 T 整除（但 nb_base 不可信）
elseif mod(nb_total, T) == 0
    H = T; nb_per = nb_total / T; mode = "infer_nb_from_T";
else
    kpi.src = "mpng:shape_unrecognized";
    obj_vdev = kpi.avg; return;
end

if H == 1
    mask = isfinite(Vm_all);
    if ~any(mask)
        kpi.src = "mpng:single_all_nan";
        obj_vdev = kpi.avg; return;
    end
    dv = Vm_all(mask) - vref;
    v = sqrt(mean(dv.^2));
    if ~isfinite(v), v = fail_pen; end
    kpi.per_hour(:) = v;   % 用同一个快照值填满 24h（用于 RL 反馈的稳定性）
    kpi.failed_hours = 0;
    kpi.avg = mean(kpi.per_hour);
    kpi.max = max(kpi.per_hour);
    kpi.used = true;
    kpi.src = "mpng:" + mode + ":nb_total=" + string(nb_total);
    obj_vdev = kpi.avg;
    return;
end

% 多时段：reshape 后逐小时计算
Vm_mat = reshape(Vm_all(1:nb_per*H), nb_per, H);

per = ones(H,1) * fail_pen;
failed = 0;
for tt = 1:H
    vcol = Vm_mat(:,tt);
    mask = isfinite(vcol);
    if ~any(mask)
        failed = failed + 1;
        per(tt) = fail_pen;
    else
        dv = vcol(mask) - vref;
        per(tt) = sqrt(mean(dv.^2));
        if ~isfinite(per(tt)), failed = failed + 1; per(tt) = fail_pen; end
    end
end

if H == T
    kpi.per_hour = per;
    kpi.failed_hours = failed;
    kpi.avg = mean(per);
    kpi.max = max(per);
    kpi.used = true;
    kpi.src = "mpng:" + mode + ":ok=" + string(H-failed) + ":fail=" + string(failed) + ":nb_per=" + string(nb_per);
    obj_vdev = kpi.avg;
else
    % 时段数不等于 T：用 H 个小时的平均值作为 24h 的代理（最小侵入）
    vbar = mean(per);
    if ~isfinite(vbar), vbar = fail_pen; end
    kpi.per_hour(:) = vbar;
    kpi.failed_hours = failed;
    kpi.avg = mean(kpi.per_hour);
    kpi.max = max(kpi.per_hour);
    kpi.used = true;
    kpi.src = "mpng:" + mode + ":inferH=" + string(H) + ":avg_to_T=" + string(T) + ":nb_per=" + string(nb_per);
    obj_vdev = kpi.avg;
end
end

function [obj_vdev, kpi] = calc_voltage_deviation_acpf(mpc_base, Pg_in, ID, T, vref, fail_pen)
% 计算 24h 电压偏差（AC PF）
% - 输入 Pg_in 来自 MOST 的 ExpectedDispatch（可能是 internal 顺序）
% - 自动映射到 external gen 行顺序；PF 失败时用 fail_pen 兜底，避免 NaN
% - 默认用 RMS(Vm-1) 作为每小时电压偏差度量

if nargin < 6 || isempty(fail_pen), fail_pen = 1.0; end
if nargin < 5 || isempty(vref),     vref     = 1.0; end
if nargin < 4 || isempty(T),        T        = size(Pg_in, 2); end

kpi = struct();
kpi.per_hour     = ones(T,1) * fail_pen;
kpi.failed_hours = 0;
kpi.avg          = fail_pen;
kpi.max          = fail_pen;
kpi.src          = "acpf:init";

% --- 1) 将 Pg 映射到 external “在线机组”顺序 ---
[idx_on_ext, Pg_on_ext, map_src] = map_Pg_to_ext_online_vdev(mpc_base, Pg_in, ID);
kpi.src = "acpf:" + string(map_src);

% --- 2) PF 选项（尽量安静 + 尽量稳定） ---
mpopt_pf = mpoption('verbose', 0, 'out.all', 0, 'pf.enforce_q_lims', 0);
% MATPOWER 8+：优先用 legacy core，避免 mp-core 的 update_z 警告刷屏
try
    mpopt_pf = mpoption(mpopt_pf, 'exp.use_legacy_core', 1);
catch
    % older MATPOWER：无该选项则忽略
end

% 先保存 warning 状态（后面恢复）
w0 = warning;

VM_COL = ID.VM;        % bus(:, VM)
BUS_TYPE_COL = 2;      % MATPOWER bus type 列固定是 2
NONE = 4;              % isolated bus type

for tt = 1:T
    mpc_t = mpc_base;

    % 写入该小时有功出力（offline 机组保持 0）
    mpc_t.gen(:, ID.PG) = 0;
    mpc_t.gen(idx_on_ext, ID.PG) = Pg_on_ext(:,tt);

    % 给所有在线机组一个统一电压设定值，减少 PV/REF 机组 setpoint 缺失引发的异常
    if isfield(ID, 'VG') && ID.VG > 0
        mpc_t.gen(idx_on_ext, ID.VG) = vref;
    end

    try
        ws = warning;                 % 只在 runpf 期间静默（不污染训练输出）
        warning('off','all');
        [rpf, success] = runpf(mpc_t, mpopt_pf);
        warning(ws);

        if ~success || ~isstruct(rpf) || ~isfield(rpf,'bus') || size(rpf.bus,2) < VM_COL
            kpi.failed_hours = kpi.failed_hours + 1;
            kpi.per_hour(tt) = fail_pen;
            continue;
        end

        Vm = rpf.bus(:, VM_COL);
        bt = rpf.bus(:, BUS_TYPE_COL);

        mask = (bt ~= NONE) & isfinite(Vm);
        if ~any(mask)
            kpi.failed_hours = kpi.failed_hours + 1;
            kpi.per_hour(tt) = fail_pen;
            continue;
        end

        % RMS 偏差（也可以换成 mean(abs(Vm-vref))）
        dv = Vm(mask) - vref;
        kpi.per_hour(tt) = sqrt(mean(dv.^2));

    catch
        % PF 异常：计入失败并兜底
        warning(w0);
        kpi.failed_hours = kpi.failed_hours + 1;
        kpi.per_hour(tt) = fail_pen;
    end
end

kpi.avg = mean(kpi.per_hour);
kpi.max = max(kpi.per_hour);

% 用 avg 作为该目标的标量（可按你的奖励设计替换）
obj_vdev = kpi.avg;
end

function [obj_vdev, kpi] = calc_voltage_deviation_acopf(mpc_base, Pg_in, ID, T, vref, fail_pen, fallback_to_pf)
% 计算 24h 电压偏差（AC OPF 优先，失败回退 PF）
% - 输入 Pg_in 来自 MOST 的 ExpectedDispatch（可能是 internal 顺序）
% - 优先使用 runopf(AC) 获取电压幅值 VM；若失败且 fallback_to_pf=true，则回退 runpf
% - 为提高可行性：将在线机组所在母线提升为 PV（REF 母线保持 REF），并允许 REF 机组吸收网损

if nargin < 7 || isempty(fallback_to_pf), fallback_to_pf = true; end
if nargin < 6 || isempty(fail_pen),       fail_pen = 1.0; end
if nargin < 5 || isempty(vref),           vref     = 1.0; end
if nargin < 4 || isempty(T),              T        = size(Pg_in, 2); end

kpi = struct();
kpi.per_hour     = ones(T,1) * fail_pen;
kpi.failed_hours = 0;
kpi.avg          = fail_pen;
kpi.max          = fail_pen;
kpi.acopf_ok_hours = 0;
kpi.pf_ok_hours   = 0;
kpi.src          = "acopf:init";

% --- 1) 将 Pg 映射到 external “在线机组”顺序 ---
[idx_on_ext, Pg_on_ext, map_src] = map_Pg_to_ext_online_vdev(mpc_base, Pg_in, ID);
kpi.src = "acopf:" + string(map_src);

% --- 2) ACOPF / ACPF 选项 ---
mpopt_acopf = mpoption('verbose', 0, 'out.all', 0, 'model', 'AC', ...
    'opf.ac.solver', 'MIPS', 'opf.start', 2, 'opf.violation', 1e-6, ...
    'pf.enforce_q_lims', 0);
mpopt_pf    = mpoption('verbose', 0, 'out.all', 0, 'pf.enforce_q_lims', 1);

% MATPOWER 8+：优先用 legacy core，避免 mp-core 的 update_z 警告刷屏
try
    mpopt_acopf = mpoption(mpopt_acopf, 'exp.use_legacy_core', 1);
    mpopt_pf    = mpoption(mpopt_pf,    'exp.use_legacy_core', 1);
catch
    % older MATPOWER：无该选项则忽略
end

% 常量（避免 define_constants 依赖）
BUS_I_COL    = 1;
BUS_TYPE_COL = 2;
PQ  = 1; PV = 2; REF = 3; NONE = 4;

VM_COL = ID.VM;     % bus(:, VM)

% 保存 warning 状态（后面恢复）
w0 = warning;

for tt = 1:T
    mpc_t = mpc_base;

    % 写入该小时有功出力（先把所有机组 PG 归零，再写入在线机组）
    mpc_t.gen(:, ID.PG) = 0;
    mpc_t.gen(idx_on_ext, ID.PG) = Pg_on_ext(:,tt);

    % 给在线机组一个统一电压设定（如果存在 VG 列）
    if isfield(ID, 'VG') && ID.VG > 0
        mpc_t.gen(idx_on_ext, ID.VG) = vref;
    end

    % --- 2.1 识别/修正 REF 母线（注意：GEN_BUS 是母线编号，不是行号） ---
    busnum = mpc_t.bus(:, BUS_I_COL);

    ref_row = find(mpc_t.bus(:, BUS_TYPE_COL) == REF, 1);
    if isempty(ref_row)
        % 如果没有 REF，选择第一个在线机组的母线作为 REF
        ref_busnum = mpc_t.gen(idx_on_ext(1), ID.GEN_BUS);
        ref_row = find(busnum == ref_busnum, 1);
        if isempty(ref_row), ref_row = 1; end
        mpc_t.bus(ref_row, BUS_TYPE_COL) = REF;
    end
    ref_busnum = mpc_t.bus(ref_row, BUS_I_COL);

    % --- 2.2 将在线机组所在母线提升为 PV（REF 母线保持 REF） ---
    gbus = unique(mpc_t.gen(idx_on_ext, ID.GEN_BUS));
    for bb = reshape(gbus, 1, [])
        brow = find(busnum == bb, 1);
        if isempty(brow), continue; end
        if mpc_t.bus(brow, BUS_TYPE_COL) == NONE, continue; end
        if brow == ref_row
            mpc_t.bus(brow, BUS_TYPE_COL) = REF;
        else
            mpc_t.bus(brow, BUS_TYPE_COL) = PV;
        end
    end

    % --- 2.3 “锁死”非 REF 机组的 Pg（REF 机组留给网损/不平衡吸收） ---
    ref_gens = find(mpc_t.gen(:, ID.GEN_STATUS) > 0 & mpc_t.gen(:, ID.GEN_BUS) == ref_busnum);

    % 给 REF 机组放宽有功上下界，便于吸收网损/不平衡（仅用于 vdev 评估）
    if ~isempty(ref_gens) && isfield(ID,'PMIN') && isfield(ID,'PMAX') && isfield(ID,'PD')
        try
            Pload = sum(mpc_t.bus(:, ID.PD));
            Pmarg = max(1.0, 0.25 * abs(Pload));
            mpc_t.gen(ref_gens, ID.PMIN) = min(mpc_t.gen(ref_gens, ID.PMIN), -Pmarg);
            mpc_t.gen(ref_gens, ID.PMAX) = max(mpc_t.gen(ref_gens, ID.PMAX),  Pload + Pmarg);
        catch
        end
    end

    fix_mask = (mpc_t.gen(:, ID.GEN_STATUS) > 0);
    fix_mask(ref_gens) = false;

    if isfield(ID,'PMIN') && isfield(ID,'PMAX')
        Pmin0 = mpc_t.gen(:, ID.PMIN);
        Pmax0 = mpc_t.gen(:, ID.PMAX);
        Pgfix = mpc_t.gen(:, ID.PG);
        Pgfix = min(max(Pgfix, Pmin0), Pmax0);

        mpc_t.gen(fix_mask, ID.PG) = Pgfix(fix_mask);
        % 窄带锁定：给 ACOPF 一点点自由度，显著提升可行性/收敛性
        p_eps = 1e-3;
        mpc_t.gen(fix_mask, ID.PMIN) = max(Pmin0(fix_mask), Pgfix(fix_mask) - p_eps);
        mpc_t.gen(fix_mask, ID.PMAX) = min(Pmax0(fix_mask), Pgfix(fix_mask) + p_eps);
    end

    % --- 3) 先尝试 ACOPF（若缺 gencost 则跳过直接回退 PF） ---
    got_vm = false;
    Vm = [];
    bt = [];

    if isfield(mpc_t, 'gencost') && ~isempty(mpc_t.gencost)
        try
            ws = warning;
            warning('off','all');
            [ropf, success] = runopf(mpc_t, mpopt_acopf);
            warning(ws);

            if success && isstruct(ropf) && isfield(ropf,'bus') && size(ropf.bus,2) >= VM_COL
                Vm = ropf.bus(:, VM_COL);
                bt = ropf.bus(:, BUS_TYPE_COL);
                got_vm = true;
                kpi.acopf_ok_hours = kpi.acopf_ok_hours + 1;
            end
        catch
            got_vm = false;
        end
    end

    % --- 4) ACOPF 失败则回退 PF ---
    if ~got_vm
        if fallback_to_pf
            try
                ws = warning;
                warning('off','all');
                [rpf, success_pf] = runpf(mpc_t, mpopt_pf);
                warning(ws);

                if success_pf && isstruct(rpf) && isfield(rpf,'bus') && size(rpf.bus,2) >= VM_COL
                    Vm = rpf.bus(:, VM_COL);
                    bt = rpf.bus(:, BUS_TYPE_COL);
                    got_vm = true;
                    kpi.pf_ok_hours = kpi.pf_ok_hours + 1;
                end
            catch
                got_vm = false;
            end
        end
    end

    if ~got_vm
        kpi.failed_hours = kpi.failed_hours + 1;
        kpi.per_hour(tt) = fail_pen;
        continue;
    end

    mask = (bt ~= NONE) & isfinite(Vm);
    if ~any(mask)
        kpi.failed_hours = kpi.failed_hours + 1;
        kpi.per_hour(tt) = fail_pen;
        continue;
    end

    dv = Vm(mask) - vref;
    kpi.per_hour(tt) = sqrt(mean(dv.^2));   % RMS 偏差
end

kpi.avg = mean(kpi.per_hour);
kpi.max = max(kpi.per_hour);


kpi.src = "acopf:" + string(map_src) + ":ok_acopf=" + string(kpi.acopf_ok_hours) + ...
          ":ok_pf=" + string(kpi.pf_ok_hours) + ":fail=" + string(kpi.failed_hours);
obj_vdev = kpi.avg;

warning(w0);
end

function [idx_on_ext, Pg_on_ext, src] = map_Pg_to_ext_online_vdev(mpc_ext, Pg_in, ID)
% 把 Pg_in 映射到 external 的 “在线机组”行顺序
% 返回：
%   idx_on_ext : external 在线机组行索引（升序）
%   Pg_on_ext  : size = [numel(idx_on_ext), T]
%   src        : 说明使用了哪种映射路径（便于 debug）

ng_ext = size(mpc_ext.gen, 1);
T      = size(Pg_in, 2);

idx_on_ext = find(mpc_ext.gen(:, ID.GEN_STATUS) > 0);
n_on_ext   = numel(idx_on_ext);

% 备用候选：假设 Pg_in 已经是 external-online 顺序
cand = {};
csrc = {};

if size(Pg_in,1) == n_on_ext
    cand{end+1} = Pg_in;
    csrc{end+1} = 'ext_online_direct';
elseif size(Pg_in,1) == ng_ext
    cand{end+1} = Pg_in(idx_on_ext, :);
    csrc{end+1} = 'ext_full_pick_online';
end

% 候选：假设 Pg_in 是 internal 顺序（MOST 常见）
try
    mpc_int = ext2int(mpc_ext);
    i2e     = mpc_int.order.gen.i2e;          % internal row -> external row
    idx_on_int = find(mpc_int.gen(:, ID.GEN_STATUS) > 0);

    Pg_ext_full = NaN(ng_ext, T);

    if size(Pg_in,1) == size(mpc_int.gen,1)
        % internal-full
        Pg_ext_full(i2e, :) = Pg_in;
        cand{end+1} = Pg_ext_full(idx_on_ext, :);
        csrc{end+1} = 'i2e_full_pick_online';
    elseif size(Pg_in,1) == numel(idx_on_int)
        % internal-online（按 idx_on_int 顺序）
        Pg_ext_full(i2e(idx_on_int), :) = Pg_in;
        cand{end+1} = Pg_ext_full(idx_on_ext, :);
        csrc{end+1} = 'i2e_online_scatter';
    end
catch
    % ext2int 不可用就跳过 internal 映射候选
end

if isempty(cand)
    % 最后兜底：截断/补零
    Pg_on_ext = zeros(n_on_ext, T);
    r = min(n_on_ext, size(Pg_in,1));
    Pg_on_ext(1:r, :) = Pg_in(1:r, :);
    src = 'fallback_trunc_zero';
    return;
end

% 用“越界次数最少”来选最合理的候选映射（自动适配 internal/external 混乱）
Pmin = mpc_ext.gen(idx_on_ext, ID.PMIN);
Pmax = mpc_ext.gen(idx_on_ext, ID.PMAX);
tol  = 1e-6;

best = 1;
best_score = inf;

for k = 1:numel(cand)
    Pg = cand{k};
    if any(isnan(Pg(:)))
        score = inf;       % 有 NaN 的直接拉黑
    else
        score = sum(Pg < (Pmin - tol) | Pg > (Pmax + tol), 'all');
    end
    if score < best_score
        best_score = score;
        best = k;
    end
end

Pg_on_ext = cand{best};
src = csrc{best};
end

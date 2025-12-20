function plot_iter_dashboard(Pg2, mpc_used, idx_wind, idx_pv1, idx_pv2, idx_gas, idx_slack, ...
                             el_boiler, pmx_wind, pmx_pv1, pmx_pv2, ...
                             cost_total, E_wind, E_pv1, E_pv2, E_gas, E_slack, E_other, ...
                             P_served, P_nonserved, Pg_storage_wind, Pg_storage_pv1,...
                             SOC_storage_wind, SOC_storage_pv1, ...
                             P_load_total, lambda_e, Pg_grid, Pg_gas)
% 根据 iter_couple_most_mpng_24h_merged 的输出数据，生成类似 Example4 的仪表盘图
if nargin < 25, Pg_gas       = []; end
if nargin < 24, Pg_grid      = []; end
if nargin < 23, lambda_e     = []; end
if nargin < 22, P_load_total = []; end
if nargin < 21, Pg_storage_pv1  = []; end
if nargin < 20, Pg_storage_wind = []; end

T = size(Pg2,2);
t = 1:T;

%% 1) “伪 UC 热图”
[UCcolor, clabels] = pseudo_uc_color(Pg2, mpc_used, idx_gas);

fh = figure('Name','Example4-like dashboard', 'Color','w', 'Units','pixels');
tiledlayout(2,4,'Padding','compact','TileSpacing','compact');

% (a) 伪 UC
% 画出 33 节点电网中所有机组的 “on gas / on non-gas” 热力图
nexttile([1 3]);
imagesc(UCcolor, 'AlphaData', 0.9);
colormap(gca, [0 0 1; 0 1 1; 1 0 0; 1 1 0]);
colorbar('Ticks',1:4,'TickLabels',clabels);
xlabel('Period (t)'); ylabel('Generator #');
title('On/Off by type (pseudo-UC)');
set(gca,'YDir','reverse');

% (b) 成本
nexttile([1 1]);
bar(1, cost_total);
set(gca,'XTick',1,'XTickLabel',{'This run'});
ylabel('Objective ($)'); title('Total Cost');

% (c) 供电/未供电
nexttile([1 1]);
bar(categorical({'Served','Non-Served'}), [P_served, P_nonserved]);
ylabel('Energy (MWh)'); title('Served vs Non-Served');

% (d) 能量构成
nexttile([1 2]);
cats = categorical({'WIND','PV1','PV2','GAS','SLACK','OTHER'});
vals = [E_wind, E_pv1, E_pv2, E_gas, E_slack, E_other];
bar(cats, vals);
ylabel('Energy (MWh)'); title('Energy by Type (24h)');

% (e) 储能出力曲线（占用第二行第4列 tile）
nexttile([1 1]);
has_storage = false;

if ~isempty(Pg_storage_wind)
    plot(t, Pg_storage_wind(:).', '-o', 'LineWidth', 1.2); hold on;
    has_storage = true;
end
if ~isempty(Pg_storage_pv1)
    plot(t, Pg_storage_pv1(:).', '-s', 'LineWidth', 1.2); hold on;
    has_storage = true;
end

if ~has_storage
    axis off;
    text(0.5, 0.5, 'no storage series', ...
        'HorizontalAlignment','center','FontAngle','italic');
else
    grid on;
    xlabel('Period (t)');
    ylabel('P_{storage} (MW)');
    lgd = {};
    if ~isempty(Pg_storage_wind), lgd{end+1} = 'Ebat-wind'; end
    if ~isempty(Pg_storage_pv1),  lgd{end+1} = 'Ebat-pv1';  end
    legend(lgd, 'Location','best');
    title('Storage power output (charge<0, discharge>0)');
end

%% 2) 储能 SOC 曲线  ===
if ~isempty(SOC_storage_wind) || ~isempty(SOC_storage_pv1)
    fh2 = figure('Name','Storage SOC','Color','w');
    hold on;
    has_soc = false;
    lgd2 = {};

    if ~isempty(SOC_storage_wind)
        plot(t, SOC_storage_wind(:).', '-o', 'LineWidth', 1.2);
        lgd2{end+1} = 'Ebat-wind SOC';
        has_soc = true;
    end
    if ~isempty(SOC_storage_pv1)
        plot(t, SOC_storage_pv1(:).', '-s', 'LineWidth', 1.2);
        lgd2{end+1} = 'Ebat-pv1 SOC';
        has_soc = true;
    end

    if has_soc
        grid on;
        xlabel('Period (t)');
        ylabel('SOC');
        ylim([0 1]);
        legend(lgd2, 'Location','best');
        title('Storage state of charge');
    else
        close(fh2);
    end
end


%% 3) 各机组类型 24 h 出力堆叠柱状图 ===
% 类型：风电、光伏、储能、燃机（或常规机组）、平衡机组、其它

[NG, T2] = size(Pg2);
if NG > 0 && T2 == T
    % 1) 先把所有机组的正向出力取出来（充电为负，不计入“发电”）
    Pg_pos = max(Pg2, 0);              % NG×T
    P_total = sum(Pg_pos, 1);          % 1×T，总发电

    % 2) 各类型索引（注意防越界和空集）
    idx_all = 1:NG;

    % 风电机组
    idx_wind_cat = [];
    if ~isempty(idx_wind)
        idx_wind_cat = idx_wind(:).';
        idx_wind_cat = idx_wind_cat(idx_wind_cat>=1 & idx_wind_cat<=NG);
    end

    % 光伏机组（PV1+PV2 合并）
    idx_pv_cat = [];
    if ~isempty(idx_pv1)
        idx_pv_cat = [idx_pv_cat, idx_pv1(:).'];
    end
    if ~isempty(idx_pv2)
        idx_pv_cat = [idx_pv_cat, idx_pv2(:).'];
    end
    idx_pv_cat = unique(idx_pv_cat);
    idx_pv_cat = idx_pv_cat(idx_pv_cat>=1 & idx_pv_cat<=NG);

    % 储能机组：通过 gentype == 'storage' 自动识别
    idx_storage_cat = [];
    if isfield(mpc_used, 'gentype') && ~isempty(mpc_used.gentype)
        gtypes = mpc_used.gentype;
        if iscell(gtypes)
            idx_storage_cat = find(strcmpi(gtypes, 'storage'));
        else
            % 兼容 char 矩阵的写法
            gtypes_cell = cellstr(strtrim(gtypes));
            idx_storage_cat = find(strcmpi(gtypes_cell, 'storage'));
        end
        idx_storage_cat = idx_storage_cat(idx_storage_cat>=1 & idx_storage_cat<=NG);
    end

    % 燃机 / 常规机组
    idx_thermal_cat = [];
    if ~isempty(idx_gas)
        idx_thermal_cat = idx_gas(:).';
        idx_thermal_cat = idx_thermal_cat(idx_thermal_cat>=1 & idx_thermal_cat<=NG);
    end

    % 平衡机组（slack）
    idx_slack_cat = [];
    if ~isempty(idx_slack)
        idx_slack_cat = idx_slack(:).';
        idx_slack_cat = idx_slack_cat(idx_slack_cat>=1 & idx_slack_cat<=NG);
    end

    % 其它机组 = 总集 - 已分类的索引
    used_idx = unique([idx_wind_cat, idx_pv_cat, idx_storage_cat, ...
        idx_thermal_cat, idx_slack_cat]);
    idx_other_cat = setdiff(idx_all, used_idx);

    % 3) 计算各类型在每个时段的正向出力和
    P_wind     = sum(Pg_pos(idx_wind_cat,   :), 1);
    P_pv       = sum(Pg_pos(idx_pv_cat,     :), 1);
    P_storage  = sum(Pg_pos(idx_storage_cat,:), 1);
    P_thermal  = sum(Pg_pos(idx_thermal_cat,:), 1);
    P_slack    = sum(Pg_pos(idx_slack_cat,  :), 1);
    P_other    = sum(Pg_pos(idx_other_cat,  :), 1);

    % 4) 组装堆叠序列（自动跳过全零类型）
    P_stack = [];
    names   = {};

    if any(P_wind > 1e-6)
        P_stack = [P_stack; P_wind];
        names{end+1} = 'Wind';
    end
    if any(P_pv > 1e-6)
        P_stack = [P_stack; P_pv];
        names{end+1} = 'PV';
    end
    if any(P_storage > 1e-6)
        P_stack = [P_stack; P_storage];
        names{end+1} = 'Storage';
    end
    if any(P_thermal > 1e-6)
        P_stack = [P_stack; P_thermal];
        names{end+1} = 'Thermal / Gas';
    end
    if any(P_slack > 1e-6)
        P_stack = [P_stack; P_slack];
        names{end+1} = 'Slack';
    end
    if any(P_other > 1e-6)
        P_stack = [P_stack; P_other];
        names{end+1} = 'Other';
    end

    % 5) 画 24 h 堆叠柱状图
    if ~isempty(P_stack)
        fh_bar = figure('Name','Generation by type (24h stacked bar)', 'Color','w');
        bar(t, P_stack.', 'stacked');          % T×Nc
        grid on;
        xlabel('Period (t)');
        ylabel('P_G (MW)');
        title('24 h generation by type (stacked bar)');
        legend(names, 'Location','eastoutside');
    end
end

%% === 4)：GRID / GAS / Storage 逐时出力 ===
if (~isempty(Pg_grid) || ~isempty(Pg_gas)) || ...
        (~isempty(Pg_storage_wind) || ~isempty(Pg_storage_pv1))

    fh_unit = figure('Name','Unit outputs (time series)','Color','w');
    hold on;
    lgd = {};

    % GRID
    if ~isempty(Pg_grid)
        plot(t, Pg_grid(:).', '-o', 'LineWidth', 1.2);
        lgd{end+1} = 'GRID';
    end

    % GAS（若不为空则视为合并后的总出力）
    if ~isempty(Pg_gas)
        plot(t, Pg_gas(:).', '-^', 'LineWidth', 1.2);
        lgd{end+1} = 'GAS';
    end

    % Storage（放在同一张图上便于看“谷充峰放”）
    if ~isempty(Pg_storage_wind)
        plot(t, Pg_storage_wind(:).', '--s', 'LineWidth', 1.2);
        lgd{end+1} = 'Storage-wind';
    end
    if ~isempty(Pg_storage_pv1)
        plot(t, Pg_storage_pv1(:).', '--d', 'LineWidth', 1.2);
        lgd{end+1} = 'Storage-pv1';
    end

    grid on;
    xlabel('Period (t)');
    ylabel('P (MW)');
    title('GRID / GAS / Storage outputs (time series)');
    if ~isempty(lgd)
        legend(lgd, 'Location','best');
    end
end

%% === 5)：负荷 vs 分时电价 λ_e（双轴） ===
if ~isempty(P_load_total) && ~isempty(lambda_e)
    fh_ld = figure('Name','Load vs time-of-use price','Color','w');

    % 左轴：系统总负荷 (MW)
    yyaxis left;
    plot(t, P_load_total(:).', '-o', 'LineWidth', 1.2);
    ylabel('Total load (MW)');
    grid on;

    % 右轴：电价标幺 λ_e 或 等效 $/MWh
    yyaxis right;
    plot(t, lambda_e(:).', '-s', 'LineWidth', 1.2);
    ylabel('\lambda_e (p.u. of base price)');

    xlabel('Period (t)');
    title('Total load and time-of-use electricity price');
end


end

clc; clear; close all;

%% 3) 构造 cap 向量
% 约定: cap = [WIND, PV1, PV2, (预留), GT上限, (预留)]
% 这里先保留 PV2、GT 等为你原来习惯的值，有需要可再调。
Cap_wind_MW = 30;
Cap_pv1_MW = 20;
Cap_pv2_MW = 20;   % 仍然用原来的默认
Cap_GT_MW  = 70;
Cap_extra  = 70;

cap = [Cap_wind_MW, Cap_pv1_MW, Cap_pv2_MW, 1.0, Cap_GT_MW, Cap_extra];

%% 5) 多目标权重（经济 / 弃风光 / 平滑）
w = [1, 1, 1, 1];

%% 6) 构造 opts 结构体
opts = struct();

% 压缩机-电网耦合设置
opts.comp_bus         = 'wind';   % 或者直接给母线号，例如 13
opts.comp_ids         = [5 7];    % 与气网 comp[05], comp[07] 对应
opts.comp_pc_unit     = 'auto';   % 'auto' | 'MW' | 'MWh/day'
opts.force_comp_as_el = true;     % 把选中的压缩机都当成电驱计入电网负荷

opts.do_plot = false;   % 算的时候不画图

%% 储能配置（示例）
% 全局往返效率（可选）
opts.storage.eta_rt = 0.9;   % 90% 往返效率，大概够现实

% 在风电母线挂一个 10 MW / 40 MWh 的电池
opts.storage.wind = struct();
opts.storage.wind.Pmax = 10;    % 充/放电功率上限 10 MW
% opts.storage.wind.bus  = 13; % 如不填就用风电机组所在母线
% opts.storage.wind.Emax = 40; % 若不填，内部自动设为 4*Pmax
opts.storage.wind.SOC0 = 0.5;   % 初始 SOC = 50%

% 在 PV1 母线挂一个 5 MW / 20 MWh 的电池
opts.storage.pv1 = struct();
opts.storage.pv1.Pmax = 10;
opts.storage.pv1.SOC0 = 0.5;

%% 7) 调用主函数（MOST + MPNG 强耦合）
out = iter_couple_most_mpng_24h_merged(cap, w, opts);
% 
% v = out.vis;
% 
% Pg2 = v.Pg2;
% 
% % GRID 单机出力（假定只有一个 GRID 机组）
% Pg_grid = [];
% if isfield(out.vis, 'idx_grid') && ~isempty(out.vis.idx_grid)
%     Pg_grid = Pg2(out.vis.idx_grid, :);
% end
% 
% % GAS 合并出力（可能有多台）
% Pg_gas = [];
% if ~isempty(out.vis.idx_gas)
%     Pg_gas = sum(Pg2(out.vis.idx_gas, :), 1);   % 1×T
% end
% plot_iter_dashboard(v.Pg2, v.mpc_used, v.idx_wind, v.idx_pv1, v.idx_pv2, ...
%     v.idx_gas, v.idx_slack, v.el_boiler, ...
%     v.pmx_wind, v.pmx_pv1, v.pmx_pv2, ...
%     out.cost_total, v.E_wind, v.E_pv1, v.E_pv2, ...
%     v.E_gas, v.E_slack, v.E_other, ...
%     v.P_served, v.P_nonserved, ...
%     out.Pg_storage_wind, out.Pg_storage_pv1, ...
%     out.SOC_storage_wind, out.SOC_storage_pv1, ...
%     out.vis.P_load, out.vis.lambda_e, ...
%     Pg_grid, Pg_gas);
% 
% % %% 画 33 节点系统的节点电压幅值和电压相角
% % plot_bus_voltage_33(out);
% 
% %% 48 节点气网的节点输出图像
% % 补充输出：
% %  1) 各气节点压力
% %  2) 各管道气体流量
% %  3) 各压缩机气体流量
% %  4) 各气井注气量 + 最大注气约束
% plot_gas_nodes_48(out);





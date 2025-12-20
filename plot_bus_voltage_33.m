function plot_bus_voltage_33(out)
% plot_bus_voltage_33
% 使用 AC 潮流(runpf)画出 33 节点系统的一次节点电压（柱状图形式）
% 输入 out 为 iter_couple_most_mpng_24h_merged 的输出结构体

% 从 out 里拿到求解时用的 mpc
if ~isfield(out, 'vis') || ~isfield(out.vis, 'mpc_used')
    error('out.vis.mpc_used 不存在，请确认 iter_couple 返回了 mpc_used。');
end
mpc = out.vis.mpc_used;

% MATPOWER 常量索引
define_constants;   % 给出 BUS_I, VM, VA 等

% AC 潮流选项（只看结果，不要一大堆命令行输出）
mpopt = mpoption('pf.alg', 'NR', ...     % 牛顿–拉夫森法
    'verbose', 0, ...
    'out.all', 0);

% 跑一次 AC 潮流
[res, success] = runpf(mpc, mpopt);
if ~success
    warning('AC 潮流未收敛，节点电压结果可能不可靠。');
end

Vm = res.bus(:, VM);   % 电压幅值 (p.u.)
Va = res.bus(:, VA);   % 电压相角 (deg)
nb = size(res.bus, 1);
buses = 1:nb;

%% 图1：电压幅值（柱状图）
figure('Name','33-bus voltage magnitude (bar)','Color','w');
bar(buses, Vm, 'FaceColor',[0.2 0.6 0.9]);
grid on;
xlabel('节点编号');
ylabel('电压幅值 / p.u.');
title('33 节点系统节点电压幅值');
xlim([0.5 nb+0.5]);
ylim([min(Vm)-0.02, max(Vm)+0.02]);

%% 图2：电压相角（柱状图）
figure('Name','33-bus voltage angle (bar)','Color','w');
bar(buses, Va, 'FaceColor',[0.9 0.4 0.2]);
grid on;
xlabel('节点编号');
ylabel('电压相角 / 度');
title('33 节点系统节点电压相角');
xlim([0.5 nb+0.5]);

end

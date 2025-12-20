function plot_gas_nodes_48(out)
% plot_gas_nodes_48 画 48 节点气网的节点“输出图像”
%
% 这里做两个层次：
%   1) node.info 的热力图：直观看每个节点各个字段的数值分布
%   2) 若存在 node.dem 或类似字段，则画每个节点“净需求 / 出力”的柱状图

if ~isfield(out, 'res_base_case') || isempty(out.res_base_case)
    error('out.res_base_case 不存在，无法取得气网结果。');
end

eg = out.res_base_case;

% 1) 抓 mgc 结构（不同版本里可能叫 mgc 或 matgas）
if isfield(eg, 'mgc') && ~isempty(eg.mgc)
    mgc = eg.mgc;
elseif isfield(eg, 'matgas') && ~isempty(eg.matgas)
    mgc = eg.matgas;
else
    error('在 res_base_case 中没有找到 mgc/matgas 字段。');
end

if ~isfield(mgc, 'node') || ~isfield(mgc.node, 'info') || isempty(mgc.node.info)
    error('mgc.node.info 不存在，无法绘制节点信息。');
end

N = double(mgc.node.info);
NB_g = size(N,1);

%% (1) node.info 热力图
figure('Name','Gas node.info heatmap','Color','w');
imagesc(N);
colorbar;
xlabel('字段索引（node.info 列）');
ylabel('气节点编号');
title(sprintf('48-node gas network: node.info (NB = %d)', NB_g));

%% (2) 若有节点需求 / 出力信息，再给一个简单柱状图
% 这里做一个尽量鲁棒的探测：优先 node.dem，其次 node.qg / node.qd 等，
% 实在没有就用 node.info 的某一列做示意。
node_val = [];

if isfield(mgc.node, 'dem') && ~isempty(mgc.node.dem)
    % 一般为 NB×1 或 NB×T，这里只取第 1 列示意
    D = double(mgc.node.dem);
    if size(D,2) > 1, D = D(:,1); end
    node_val = D;
end

% 如有其它字段，可在这里按你自己气网的格式再加几种备选：
%   if isfield(mgc.node, 'load'), node_val = double(mgc.node.load(:,1)); end

if isempty(node_val)
    % 退而求其次：用 node.info 每行的和/某列值作为“代表输出”
    node_val = sum(N, 2);
end

figure('Name','Gas node aggregate value','Color','w');
bar(1:NB_g, node_val);
xlabel('气网节点编号');
ylabel('总用气量 / (10^4 Nm^3)');
title('48-node gas network: per-node aggregate value');
grid on;

% 仿 MPNG Example2 的四种图形：
%  1) 各气节点压力
%  2) 各管道气体流量
%  3) 各压缩机气体流量
%  4) 各气井注气量 + 最大注气约束
%
% 输入：
%   res_gas   – 一次 MPNG 运行的结果结构体（你的 out.res_base_case）
%   run_label – 字符串，用于图题和图例，例如 'Coupled 33-bus & 48-node case'
% MPNG 自带的 gas 常数（PR、F_NODE、T_NODE、FG_O、FG_C、G、GMAX 等）
define_constants_gas;

% 为兼容不同版本：有的结果叫 mgc，有的叫 matgas
res_gas = out.res_base_case;
run_label = 'Coupled case';
if isfield(res_gas, 'mgc')
    mgc = res_gas.mgc;
elseif isfield(res_gas, 'matgas')
    mgc = res_gas.matgas;
else
    error('结果结构体中找不到 mgc/matgas 字段。');
end

%% 1) 各气节点压力（对应 Example2 的 Figure 1）
p = mgc.node.info(:, PR);              % 节点压力 [psia]
x = 1:numel(p);                        % 节点编号

figure('Name', 'Gas nodal pressure');
stem(x, p, 'filled');
grid on; box on;
xlabel('Node');
ylabel('Pressure [psia]');
title(['Nodal Pressure - ' run_label]);
xlim([0, numel(p)+1]);

%% 2) 各管道气体流量（对应 Example2 的 Figure 2）
fpipe   = mgc.pipe(:, F_NODE);        % 起点节点号
tpipe   = mgc.pipe(:, T_NODE);        % 终点节点号
tag_pipe = cell(length(fpipe), 1);
for k = 1:length(fpipe)
    tag_pipe{k} = sprintf('%d-%d', fpipe(k), tpipe(k));
end

fgo = mgc.pipe(:, FG_O);              % 管道流量 [MMSCFD]

figure('Name', 'Gas flow in pipelines');
bar(1:length(fgo), fgo);
grid on; box on;
xlabel('Pipeline');
ylabel('Gas Flow [MMSCFD]');
title(['Gas Flow in All Pipelines - ' run_label]);
set(gca, 'XTick', 1:length(fpipe));
set(gca, 'XTickLabel', tag_pipe);
xtickangle(90);

%% 3) 各压缩机气体流量（对应 Example2 的 Figure 3）
if ~isempty(mgc.comp)
    fcomp   = mgc.comp(:, F_NODE);
    tcomp   = mgc.comp(:, T_NODE);
    tag_comp = cell(length(fcomp), 1);
    for k = 1:length(fcomp)
        tag_comp{k} = sprintf('%d-%d', fcomp(k), tcomp(k));
    end

    fgc = mgc.comp(:, FG_C);          % 压缩机气体流量 [MMSCFD]

    figure('Name', 'Gas flow in compressors');
    bar(1:length(fgc), fgc);
    grid on; box on;
    xlabel('Compressor');
    ylabel('Gas Flow [MMSCFD]');
    title(['Gas Flow in All Compressors - ' run_label]);
    set(gca, 'XTick', 1:length(fcomp));
    set(gca, 'XTickLabel', tag_comp);
else
    warning('mgc.comp 为空，本次没有压缩机流量图。');
end

%% 4) 各气井注气量（对应 Example2 的 Figure 4）
if ~isempty(mgc.well)
    g     = mgc.well(:, G);           % 实际注气量 [MMSCFD]
    gmax  = mgc.well(:, GMAX);        % 最大注气 [MMSCFD]
    x     = 1:length(g);

    figure('Name', 'Gas injections at wells');
    hold on; grid on; box on;
    bar(x, g);                        % 柱状图：实际注气
    plot(x, gmax, '*r', 'MarkerSize', 8);   % 星号：最大注气约束
    xlabel('Gas Wells');
    ylabel('Gas Injection [MMSCFD]');
    title(['Gas Injections at All Wells - ' run_label]);
    legend({run_label, 'Max. injection'}, 'Location', 'best');
    xlim([0.5, length(x)+0.5]);
else
    warning('mgc.well 为空，本次没有气井注气图。');
end

end

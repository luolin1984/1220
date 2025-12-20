function plot_uc_allgens(out)
% plot_uc_allgens 画出 33 节点电网中所有机组的 “on gas / on non-gas” 热力图
%
% 输入：out = iter_couple_most_mpng_24h_merged 的输出结构体

v       = out.vis;
Pg2     = v.Pg2;          % NG × T 发电功率矩阵（内部顺序）
mpc     = v.mpc_used;     % 求解时用到的 mpc
idx_gas = v.idx_gas;      % 燃气机组在 mpc.gen 中的行号

[UCcolor, clabels] = pseudo_uc_color_local(Pg2, mpc, idx_gas);

T = size(Pg2, 2);

figure('Name','UC on/off by type (all gens)','Color','w');
imagesc(1:T, 1:size(UCcolor,1), UCcolor, 'AlphaData', 0.9);
colormap([0 0 1; 0 1 1; 1 0 0; 1 1 0]);   % 蓝:off non-gas; 青:on non-gas; 红:off gas; 黄:on gas
colorbar('Ticks',1:4,'TickLabels',clabels);

set(gca,'YDir','reverse');
xlabel('Period (t)');
ylabel('Generator # (ext index)');
title('On/Off by type (all generators)');

end

% ===== 本文件内部的小工具：与 iter_couple 里同名函数逻辑一致 =====
function [UCcolor, clabels] = pseudo_uc_color_local(Pg2, mpc, idx_gas)
% 用 Pg>eps 近似“上线=1/离线=0”；气 / 非气四色

eps_on = 1e-6;
ng = size(mpc.gen,1);
T  = size(Pg2,2);

UC = zeros(ng, T);
try
    % 这里只是为了保持行数一致，不做复杂映射
    [~, ord] = ext2int(mpc); %#ok<ASGLU>
catch
end

for gi = 1:ng
    UC(gi,:) = Pg2(gi,:) > eps_on;
end

mask_gas = false(ng,1);
if ~isempty(idx_gas) && all(idx_gas >= 1) && all(idx_gas <= ng)
    mask_gas(idx_gas) = true;
end

UCcolor = ones(size(UC));            % 默认 off non-gas = 1
UCcolor(UC==1 & ~mask_gas) = 2;      % on non-gas
UCcolor(UC==0 &  mask_gas) = 3;      % off gas
UCcolor(UC==1 &  mask_gas) = 4;      % on gas

clabels = {'off non-gas','on non-gas','off gas','on gas'};
end

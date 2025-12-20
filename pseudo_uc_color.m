function [UCcolor, clabels] = pseudo_uc_color(Pg2, mpc, idx_gas)
% 用 Pg>eps 近似“上线=1/离线=0”；气/非气双色
eps_on = 1e-6;
ng = size(mpc.gen,1);
T  = size(Pg2,2);
UC = zeros(ng, T);
try
    [~, ord] = ext2int(mpc); i2e = ord.gen.i2e; % 为了和外部编号一致仅做同形矩阵
catch
    i2e = (1:ng).';
end
for gi = 1:ng
    UC(gi,:) = Pg2(gi,:) > eps_on;   % 每个时段独立判断
end
mask_gas = false(ng,1);
if ~isempty(idx_gas) && idx_gas>=1 && idx_gas<=ng
    mask_gas(idx_gas) = true;
end
% 映射成 1..4 的彩色标签（与 Example4 同口径）
UCcolor = ones(size(UC));
UCcolor(UC==1 & ~mask_gas) = 2;  % on non-gas
UCcolor(UC==0 &  mask_gas) = 3;  % off gas
UCcolor(UC==1 &  mask_gas) = 4;  % on gas
clabels = {'off non-gas','on non-gas','off gas','on gas'};
end
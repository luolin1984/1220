mgc = out.res_base_case.mgc;
define_constants_gas;   % DEM, PR, PRMAX, PRMIN, FG_O, FMAX_O, ...

%% 1) 节点压力情况
P    = [];
Pmin = [];
Pmax = [];

if isfield(mgc,'node')
    if isstruct(mgc.node)
        % 常见形式：mgc.node.info 是状态量，mgc.node 本体/ mgc.node.data 是参数
        Ninfo = [];
        if isfield(mgc.node,'info')
            Ninfo = mgc.node.info;
        end

        % 参数表：有些版本放在 node.data，有些直接把上下限放在 node.info 里
        if isfield(mgc.node,'data')
            Ntab = mgc.node.data;
        else
            Ntab = Ninfo;   % 退一步：当成在 info 里
        end
    else
        % 老版本：mgc.node 就是数值矩阵
        Ntab  = mgc.node;
        Ninfo = [];
    end

    if ~isempty(Ntab)
        % 实际压力：优先 info(PR)，否则 tab(PR)
        if ~isempty(Ninfo) && size(Ninfo,2) >= PR
            P = Ninfo(:, PR);
        else
            P = Ntab(:, PR);
        end

        % 上下限：一般在参数表里
        if size(Ntab,2) >= PRMIN
            Pmin = Ntab(:, PRMIN);
            Pmax = Ntab(:, PRMAX);
        end
    end
end

fprintf('--- Node pressure debug ---\n');
if ~isempty(P)
    fprintf('P    range = [%.3f, %.3f]\n', min(P),   max(P));
else
    fprintf('P    为空（没找到 PR 列）\n');
end
if ~isempty(Pmin)
    fprintf('Pmin range = [%.3f, %.3f]\n', min(Pmin), max(Pmin));
end
if ~isempty(Pmax)
    fprintf('Pmax range = [%.3f, %.3f]\n', min(Pmax), max(Pmax));
end

n = min([numel(P), numel(Pmin), numel(Pmax)]);
P    = P(1:n);
Pmin = Pmin(1:n);
Pmax = Pmax(1:n);

viol = find(P > Pmax + 1e-3);
fprintf('pressure violation nodes = %d\n', numel(viol));

if ~isempty(viol)
    disp([ (1:n).', P, Pmax ])  % 看一眼是哪几个点越限了
end

%% 2) 管道流量 / 上限情况
F    = [];
Fmax = [];

if isfield(mgc,'pipe') && ~isempty(mgc.pipe)
    Ptab = mgc.pipe;

    if isnumeric(Ptab)
        % 数值矩阵版本
        if size(Ptab,2) >= FG_O
            F = Ptab(:, FG_O);
        end
        if size(Ptab,2) >= FMAX_O
            Fmax = Ptab(:, FMAX_O);
        end
    elseif isstruct(Ptab)
        % struct 版本：参数在 data，结果在 info
        if isfield(Ptab,'data')
            D = Ptab.data;
            if size(D,2) >= FMAX_O
                Fmax = D(:, FMAX_O);
            end
        end
        if isfield(Ptab,'info')
            I = Ptab.info;
            if size(I,2) >= FG_O
                F = I(:, FG_O);
            end
        end
    end
end

fprintf('--- Pipe flow debug ---\n');
if ~isempty(F)
    fprintf('F     range = [%.3e, %.3e]\n', min(F),    max(F));
else
    fprintf('F     为空（没找到 FG_O 列）\n');
end
if ~isempty(Fmax)
    fprintf('Fmax  range = [%.3e, %.3e]\n', min(Fmax), max(Fmax));
end

if ~isempty(F) && ~isempty(Fmax)
    n = min(numel(F), numel(Fmax));
    loading = abs(F(1:n)) ./ max(Fmax(1:n), 1e-6);
    fprintf('loading range = [%.3f, %.3f]\n', min(loading), max(loading));
end

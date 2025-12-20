function UCcolor = uc2image(res)
define_constants
define_constants_gas
id_gas_gen = res.connect.interc.term(:,GEN_ID);
nt = size(res.connect.power.time,2);
ng = size(res.genid.original,1)/nt;
if isempty(res.connect.power.UC)
    res.connect.power.UC = ones(ng,nt);
end
mask_gas_gen = zeros(ng,1);
mask_gas_gen(id_gas_gen) = 1;
mask_gas_gen = repmat(mask_gas_gen,1,nt);
UCcolor = ones(size(mask_gas_gen));
UCcolor(res.connect.power.UC == 1 & mask_gas_gen == 1) = 4;
UCcolor(res.connect.power.UC == 0 & mask_gas_gen == 1) = 3;
UCcolor(res.connect.power.UC == 1 & mask_gas_gen == 0) = 2;
UCcolor(res.connect.power.UC == 0 & mask_gas_gen == 0) = 1;
end
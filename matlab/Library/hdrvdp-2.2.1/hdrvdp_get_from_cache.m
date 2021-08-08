function var = hdrvdp_get_from_cache( name, key, func )

persistent hdrvdp_cache;

if( ~isfield( hdrvdp_cache, name ) || any(hdrvdp_cache.(name).key ~= key) )
    % Cache does not exist or needs updating

    hdrvdp_cache.(name) = struct();
    hdrvdp_cache.(name).key = key;
    var = func();
    hdrvdp_cache.(name).var = var;    
else
    % Data can be fetched from the cache
    var = hdrvdp_cache.(name).var;
end

end

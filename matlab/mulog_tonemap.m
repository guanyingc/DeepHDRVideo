function tonemap_hdr = mulog_tonemap(hdr, mu)
    denom = log(1.0 + mu);
    tonemap_hdr = log(1.0 + hdr .* mu) ./ denom;
end

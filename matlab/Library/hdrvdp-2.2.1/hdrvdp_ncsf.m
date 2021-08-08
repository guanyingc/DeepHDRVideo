function S = hdrvdp_ncsf( rho, lum, metric_par )
% Neural contrast sensitivity function
%
% S = hdrvdp_csf( rho, lum, metric_par )
%
% This is a naural contrast sensitivity function, which does not account
% for the optical component, nor luminance-dependent component. To compute
% a complete CSF, use: 
%
% CSF = hdrvdp_csf( rho, lum, metric_par ) * hdrvdp_mtf( rho, metric_par ) *
%    hdrvdp_joint_rod_cone_sens( lum, metric_par );
%
% Note that the peaks of nCSF are not normalized to 1. This is to account
% for the small variations in the c.v.i. (sensitivity due to adapting
% luminance). 
%
% Copyright (c) 2011, Rafal Mantiuk <mantiuk@gmail.com>

% Permission to use, copy, modify, and/or distribute this software for any
% purpose with or without fee is hereby granted, provided that the above
% copyright notice and this permission notice appear in all copies.
%
% THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
% WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
% MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
% ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
% WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
% ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
% OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

csf_pars = metric_par.csf_params;
lum_lut = log10(metric_par.csf_lums);

log_lum = log10( lum );

par = zeros(length(lum),4);
for k=1:4
    par(:,k) = interp1( lum_lut, csf_pars(:,k+1), clamp( log_lum, lum_lut(1), lum_lut(end) ) );
end

S =  par(:,4) .* 1./((1+(par(:,1).*rho).^par(:,2)) .* 1./(1-exp(-(rho/7).^2)).^par(:,3)).^0.5;
end
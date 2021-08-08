function S = hdrvdp_rod_sens( la, metric_par )
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


S = zeros( size( la ) );

peak_l = metric_par.csf_sr_par(1);
low_s = metric_par.csf_sr_par(2);
low_exp = metric_par.csf_sr_par(3);
high_s = metric_par.csf_sr_par(4);
high_exp = metric_par.csf_sr_par(5);
rod_sens = metric_par.csf_sr_par(6);


ss = la>peak_l; 
S(ss) = exp( -abs(log10(la(ss)/peak_l)).^high_exp/high_s );
S(~ss) = exp( -abs(log10(la(~ss)/peak_l)).^low_exp/low_s );

S = S * 10.^rod_sens; %TODO: check if this is really needed

end
function Sd = hdrvdp_joint_rod_cone_sens_diff( la, metric_par )
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

p1 = metric_par.csf_sa(1);
p2 = metric_par.csf_sa(2); % in the paper - p6
p3 = metric_par.csf_sa(3); % in the paper - p7
p4 = metric_par.csf_sa(4); % in the paper - p8

Sd = (p1*p2*p3*p4*(p2./la).^(p3 - 1))./(la.^2.*((p2./la).^p3 + 1).^(p4 + 1));

end
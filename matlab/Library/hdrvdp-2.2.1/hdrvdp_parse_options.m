function metric_par = hdrvdp_parse_options( options )
% HDRVDP_PARSE_OPTIONS (internal) parse HDR-VDP options and create two
% structures: view_cond with viewing conditions and metric_par with metric
% parameters
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

% Defaults

metric_par.debug = false;

% Peak contrast from Daly's CSF for L_adapt = 30 cd/m^2
daly_peak_contrast_sens = 0.006894596;

%metric_par.sensitivity_correction = daly_peak_contrast_sens / 10.^-2.355708; 
metric_par.sensitivity_correction = daly_peak_contrast_sens / 10.^-2.4; 

metric_par.view_dist = 0.5;

metric_par.spectral_emission = [];


metric_par.orient_count = 4; % the number of orientations to consider

% Various optional features
metric_par.do_masking = true;
metric_par.do_mtf = true;
metric_par.do_spatial_pooling = true;
metric_par.noise_model = true;
metric_par.do_quality_raw_data = false; % for development purposes only
metric_par.do_si_gauss = false;
metric_par.si_size = 1.008;

% Warning messages
metric_par.disable_lowvals_warning = false;

metric_par.steerpyr_filter = 'sp3Filters';

metric_par.mask_p = 0.544068;
metric_par.mask_self = 0.189065;
metric_par.mask_xo = 0.449199;
metric_par.mask_xn = 1.52512;
metric_par.mask_q = 0.49576;
metric_par.si_size = -0.034244;

metric_par.psych_func_slope = log10(3.5);
metric_par.beta = metric_par.psych_func_slope-metric_par.mask_p;

% Spatial summation
metric_par.si_slope = -0.850147;
metric_par.si_sigma = -0.000502005;
metric_par.si_ampl = 0;

% Cone and rod cvi functions
metric_par.cvi_sens_drop = 0.0704457;
metric_par.cvi_trans_slope = 0.0626528;
metric_par.cvi_low_slope = -0.00222585;

metric_par.rod_sensitivity = 0;
%metric_par.rod_sensitivity = -0.383324;
metric_par.cvi_sens_drop_rod = -0.58342;

% Achromatic CSF
metric_par.csf_m1_f_max = 0.425509;
metric_par.csf_m1_s_high = -0.227224;
metric_par.csf_m1_s_low = -0.227224;
metric_par.csf_m1_exp_low = log10( 2 );


metric_par.csf_params = [ ...
   0.0160737   0.991265   3.74038   0.50722   4.46044
   0.383873   0.800889   3.54104   0.682505   4.94958
   0.929301   0.476505   4.37453   0.750315   5.28678
   1.29776   0.405782   4.40602   0.935314   5.61425
   1.49222   0.334278   3.79542   1.07327   6.4635
   1.46213   0.394533   2.7755   1.16577   7.45665 ];

metric_par.csf_lums = [ 0.002 0.02 0.2 2 20 150];

metric_par.csf_sa = [30.162 4.0627 1.6596 0.2712];

metric_par.csf_sr_par = [1.1732 1.1478 1.2167 0.5547 2.9899 1.1414]; % rod sensitivity function

par = [0.061466549455263 0.99727370023777070]; % old parametrization of MTF
metric_par.mtf_params_a = [par(2)*0.426 par(2)*0.574 (1-par(2))*par(1) (1-par(2))*(1-par(1))];
metric_par.mtf_params_b = [0.028 0.37 37 360];

%metric_par.quality_band_freq = [15 7.5 3.75 1.875 0.9375 0.4688 0.2344];
metric_par.quality_band_freq = [60 30 15 7.5 3.75 1.875 0.9375 0.4688 0.2344 0.1172];

%metric_par.quality_band_w = [0.2963    0.2111    0.1737    0.0581   -0.0280    0.0586    0.2302];

% New quality calibration: LDR + HDR datasets - paper to be published
%metric_par.quality_band_w = [0.2832    0.2142    0.2690    0.0398    0.0003    0.0003    0.0002];
metric_par.quality_band_w = [0 0.2832 0.2832    0.2142    0.2690    0.0398    0.0003    0.0003    0 0];

metric_par.quality_logistic_q1 = 3.455;
metric_par.quality_logistic_q2 = 0.8886;

metric_par.calibration_date = '30 Aug 2011';

metric_par.surround_l = 1e-5; 

% process options
i = 1;
while( i <= length( options ) )
    if( strcmp( options{i}, 'pixels_per_degree' ) )
        i = i+1;
        metric_par.pix_per_deg = options{i};
    elseif( strcmp( options{i}, 'viewing_distance' ) )
        i = i+1;
        metric_par.view_dist = options{i};
    elseif( strcmp( options{i}, 'peak_sensitivity' ) )
        i = i+1;
        metric_par.sensitivity_correction = daly_peak_contrast_sens / 10.^(-options{i});
    else
        % all other options
        metric_par.(options{i}) = options{i+1};
        i = i+1;
    end
    i = i+1;
end

end
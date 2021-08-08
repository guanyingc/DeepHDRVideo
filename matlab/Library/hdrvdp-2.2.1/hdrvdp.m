function res = hdrvdp( test, reference, color_encoding, pixels_per_degree, options )
% HDR-VDP-2 compute visually significant differences between an image pair.
%
% diff = HDRVDP( test, reference, color_encoding, pixels_per_degree, options )
%
% Parameters:
%   test - image to be tested (e.g. with distortions)
%   reference - reference image (e.g. without distortions)
%   color_encoding - color representation for both input images. See below.
%   pixels_per_degree - visual resolution of the image. See below.
%   options - cell array with { 'option', value } pairs. See the list of options
%       below. Note if unknown or misspelled option is passed, no warning
%       message is issued.
%
% The function returns a structure with the following fields:
%   P_map - probability of detection per pixel (matrix 0-1)
%   P_det - a single valued probability of detection (scalar 0-1)
%   C_map - threshold normalized contrast map, so that C_max=1
%           corresponds to the detection threshold (P_det=0.5).
%   C_max - maximum threshold normalized contrast, so that C_max=1
%           corresponds to the detection threshold (P_det=0.5).
%   Q     - Quality correlate, which is 100 for the best quality and gets 
%           lower for lower quality. Q can be negative in case of very large 
%           differences.            
%
% Test and references images are matrices of size (height, width,
% channel_count). If there is only one channel, it is assumed to be
% achromatic (luminance) with D65 spectra. The values must be given in
% absolute luminance units (cd/m^2). If there are three color channels,
% they are assumed to be RGB of an LCD display with red, green and blue LED
% backlight. If different number of color channels is passed, their spectral
% emission curve should be stored in a comma separated text file and its
% name should be passed as 'spectral_emission' option.
%
% Note that the current version of HDR-VDP does NOT take color differences
% into account. Spectral channels are used to properly compute luminance
% sensed by rods and cones.
%
% COLOR ENCODING:
%
% HDR-VDP-2 requires that the color encoding is specified explicitly to avoid
% mistakes when passing images. HDR-VDP operates on absolue physical units,
% not pixel values that can be found in images. Therefore, it is necessary
% to specify how the metric should interpret input images. The available
% options are:
%
% 'luminance' - images contain absolute luminance values provided in
% photopic cd/m^2. The images must contain exactly one color channel.
%
% 'luma-display' - images contain grayscale pixel values, sometimes known
% as gamma-corrected luminance. The images must contain exactly one color
% channel and the maximum pixel value must be 1 (not 256).
% It corresponds to a gray-scale channel in
% YCrCb color spaces used for video encoding. Because 'luma' alone does not
% specify luminance, HDR-VDP-2 assumes the following display model:
%
% L = 99 * V^2.2 + 1,
%
% where L is luminance and V is luma. This corresponds to a display with
% 2.2 gamma, 100 cd/m^2 maximum luminance and 1 cd/m^2 black level.
%
% 'sRGB-display' - use this color encoding for standard (LDR) color images.
% This encoding assumes an sRGB display, with 100 cd/m^2 peak luminance and
% 1 cd/m^2 black level. Note that this is different from sRGB->XYZ
% transform, which assumes the peak luminance of 80 cd/m^2 and 1 cd/m^2
% black level. The maximum pixel value must be 1 (not 256).
%
% 'rgb-bt.709' - use this color space for HDR images AFTER they are at
% least roughly calibrated in absolute photometric units. The encoding
% assumes the ITU-R BT.709 RGB color primaries (the same as for sRGB), but also
% non-gamma corrected RGB color space.
%
% 'XYZ' - input image is provided as ABSOLUTE trichromatic color values in
% CIE XYZ (1931) color space. The Y channel must be equal luminance in
% cd/m^2.
%
% PIXELS_PER_DEGREE:
%
% This argument specifies the angular resolution of the image in terms of the
% number of pixels per one visual degree. It
% will change with a viewing distance and display resolution. A typical
% value for a stanard resolution computer display is around 30. A
% convenience function hdrvdp_pix_per_deg can be used to compute pixels per
% degree parameter.
%
% OPTIONS:
% The options must be given as name-value pairs in a cell array.
% Default values are given in square brackets.
%
%   'surround_l' - [mean value of each color channel] luminance/intensity of the
%      surround, which is assumed to be uniform outside the input images.
%      The default value is 1e-5 (10^-5), which corresponds to almost
%      absolute darkness. Note that surround_l=0 should be avoided. It is 
%      unrealistic to expect absolutely no light in physically plausible 
%      environment. The special value -1 makes the surround surround_l
%      equal to the geometric mean of the image.
%   'spectral_emission' - name of the comma separated file that contains
%      spectral emission curves of color channels of reference and test
%      images.
%   'rgb_display' - [ccfl-lcd] use this option to specify one of the
%   predefined emission spectra for typical displays. Availaeble options
%   are:
%        crt - a typical CRT display
%        ccfl-lcd - an LCD display with CCFL backlight
%        led-lcd - an LCD display with LED backlight
%
% The following are the most important options used to fine-tune and calibrate
% HDR-VDP:
%   'peak_sensitivity' - absolute sensitivity of the HDR-VDP
%   'mask_p' - excitation of the visual contrast masking model
%   'mask_q' - inhibition of the visual contrast masking model
%
% EXAMPLE:
% The following example creates a luminance ramp (gradient), distorts it
% with a random noise and computes detection probabilities using HDR-VDP.
%
% reference = logspace( log10(0.1), log10(1000), 512 )' * ones( 1, 512 );
% test = reference .* (1 + (rand( 512, 512 )-0.5)*0.1);
% res = hdrvdp( test, reference, 'luminance', 30, { 'surround_l', 13 } );
% clf;
% imshow( hdrvdp_visualize( res.P_map, test ) );
%
% BUGS and LIMITATIONS:
%
% If you suspect the predictions are wrong, check first the Frequently
% Asked Question section on the HDR-VDP-2 web-site
% (http://hdrvdp.sourceforge.net). If it does not help, post your problem to 
% the HDR-VDP discussion group: http://groups.google.com/group/hdrvdp 
% (preffered) or send an e-mail directly to the author.
%
% REFERENCES
% 
% The metric is described in the paper: 
% Mantiuk, Rafat, Kil Joong Kim, Allan G. Rempel, and Wolfgang Heidrich.
% ?HDR-VDP-2: A Calibrated Visual Metric for Visibility and 
%  Quality Predictions in All Luminance Conditions.? 
% ACM Trans. Graph (Proc. SIGGRAPH) 30, no. 4 (July 01, 2011): 1. 
% doi:10.1145/2010324.1964935.
% 
% When refering to the metric, please cite the paper above and include the
% version number, for example "HDR-VDP 2.2.0". To check the version number,
% see the ChangeLog.txt. To check the version in the code, call
% hdrvdp_version.txt.
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

if( any( size(test) ~= size(reference) ) )
    error( 'reference and test images must be the same size' );
end

if( ~exist( 'options', 'var' ) )
    options = {};
end


if( ~exist( 'reconSpyr', 'file' ) )
    % If matlabPyrTools not in the path, add them now
    
    % Get the path to the hdrvdp directory
    [pathstr, name, ext, versn] = fileparts(mfilename( 'fullpath' ));
    
    addpath( fullfile( pathstr, 'matlabPyrTools_1.4_fixed' ) );
    
    % Re-check if everything went OK
    if( ~exist( 'reconSpyr', 'file' ) )
        error( 'Failed to add matlabPyrTools to the path.' );
    end
end

metric_par = hdrvdp_parse_options( options );

% The parameters overwrite the options
if( ~isempty( pixels_per_degree ) )
    metric_par.pix_per_deg = pixels_per_degree;
end
if( ~isempty( color_encoding ) )
    metric_par.color_encoding = color_encoding;
end

% Load spectral emission curves
img_channels = size( test, 3 );

switch lower( metric_par.color_encoding )
    case 'luminance'
        if( img_channels ~= 1 )
            error( 'Only one channel must be provided for "luminance" color encoding' );
        end
        check_if_values_plausible( test, metric_par );
    case 'luma-display'
        if( img_channels ~= 1 )
            error( 'Only one channel must be provided for "luma-display" color encoding' );
        end
        test = display_model( test, 2.2, 99, 1 );
        reference = display_model( reference, 2.2, 99, 1 );
    case 'srgb-display'
        if( img_channels ~= 3 )
            error( 'Exactly 3 color channels must be provided for "sRGB-display" color encoding' );
        end
        test = display_model_srgb( test );
        reference = display_model_srgb( reference );        
    case 'rgb-bt.709'
        if( img_channels ~= 3 )
            error( 'Exactly 3 color channels must be provided for "rgb-bt.709" color encoding' );
        end
        check_if_values_plausible( test(:,:,2), metric_par );
    case 'xyz'
        if( img_channels ~= 3 )
            error( 'Exactly 3 color channels must be provided for "XYZ" color encoding' );
        end
        test = xyz2rgb( test );
        reference = xyz2rgb( reference );
        check_if_values_plausible( test(:,:,2), metric_par );
    case 'generic'
        if( isempty( metric_par.spectral_emission ) )
            error( '"spectral_emission" option must be specified when using the "generic" color encoding' );
        end
end


if( metric_par.surround_l == -1 )
    % If surround_l is set to -1, use the geometric (log) mean of each color
    % channel of the reference image
    metric_par.surround_l = geomean( reshape( reference, [size(reference,1)*size(reference,2) size(reference,3)] ) );
elseif( length(metric_par.surround_l) == 1 )
    metric_par.surround_l = repmat( metric_par.surround_l, [1 img_channels] );
end

if( img_channels == 3 && ~isfield( metric_par, 'rgb_display' ) )
    metric_par.rgb_display = 'ccfl-lcd';
end

if( ~isempty( metric_par.spectral_emission ) )
    [tmp, IMG_E] = load_spectral_resp( [metric_par.base_dir '/' metric_par.spectral_emission] );
elseif( isfield( metric_par, 'rgb_display' ) )
    [tmp, IMG_E] = load_spectral_resp( sprintf( 'emission_spectra_%s.csv', metric_par.rgb_display ) );
elseif( img_channels == 1 )
    [tmp, IMG_E] = load_spectral_resp( 'd65.csv' );
else
    error( '"spectral_emissiom" option needs to be specified' );
end

if( img_channels == 1 && size(IMG_E,2)>1 )
    % sum-up spectral responses of all channels for luminance-only data
    IMG_E = sum( IMG_E, 2 );
end

if( img_channels ~= size( IMG_E, 2 ) )
    error( 'Spectral response data is either missing or is specified for different number of color channels than the input image' );
end
metric_par.spectral_emission = IMG_E;

% Compute spatially- and orientation-selective bands
% Process reference first to reuse bb_padvalue
[B_R L_adapt_reference band_freq bb_padvalue] = hdrvdp_visual_pathway( reference, 'reference', metric_par, -1 );
[B_T L_adapt_test] = hdrvdp_visual_pathway( test, 'test', metric_par, bb_padvalue );

L_adapt = (L_adapt_test + L_adapt_reference)./2;

% precompute CSF
csf_la = logspace( -5, 5, 256 );
csf_log_la = log10( csf_la );
b_count = length(B_T.sz);

CSF = zeros( length(csf_la), b_count );
for b=1:b_count
    CSF(:,b) = hdrvdp_ncsf( band_freq(b), csf_la', metric_par );
end

log_La = log10(clamp( L_adapt, csf_la(1), csf_la(end) ));

D_bands = B_T;

% Pixels that are actually different
diff_mask = any((abs( test-reference ) ./ reference) > 0.001, 3);

if( metric_par.do_quality_raw_data )
    qres = quality_pred_init();
end

Q = 0; % quality correlate

% For each band
for b = 1:b_count
    
    %masking params
    p = 10.^metric_par.mask_p;
    q = 10.^metric_par.mask_q;
    pf = (10.^metric_par.psych_func_slope)/p;
        
    % accumulate masking activity across orientations (cross-orientation
    % masking)
    mask_xo = zeros( get_band_size(B_T,b,1) );
    for o=1:B_T.sz(b)
        mask_xo = mask_xo + mutual_masking( b, o ); %min( abs(B_T{b,o}), abs(B_R{b,o}) );
    end
    
    log_La_rs = clamp( imresize(log_La,get_band_size(B_T,b,1)), csf_log_la(1), csf_log_la(end) );
    % per-pixel contrast sensitivity
    CSF_b = interp1( csf_log_la, CSF(:,b), log_La_rs );
    
    % REMOVED: Transform CSF linear sensitivity to the non-linear space of the
    % photoreceptor response
%    CSF_b = CSF_b .* 1./hdrvdp_joint_rod_cone_sens_diff( 10.^log_La_rs, metric_par );
        
    band_norm = 2^(b-1); % = 1/n_f from the paper
    band_mult = 1; %2^-(b-1);
    
    for o=1:B_T.sz(b)
        
        %excitation difference
        band_diff = (get_band(B_T,b,o) - get_band(B_R,b,o))*band_mult;
        
        if( metric_par.do_si_gauss )            
            band_scale = size(band_diff,1)/size(test,1);
            % Sigma grows with lower frequencies to subtend a similar number of
            % cycles. Note that the scale differs between the bands.
            sigma = 10^metric_par.si_size / band_freq(b) * metric_par.pix_per_deg * band_scale;
            local_sum = fast_gauss( abs(band_diff), sigma, false );
            ex_diff = (sign_pow(band_diff/band_norm, p-1) * band_norm) .* local_sum;
        else
            ex_diff = sign_pow(band_diff/band_norm, p) * band_norm;
        end
        
        
        if( b == b_count )
            % base band
            N_nCSF = 1;
        else
            N_nCSF = (1./CSF_b);
        end
        
        if( metric_par.do_masking )
            
            k_mask_self = 10.^metric_par.mask_self; % self-masking
            k_mask_xo = 10.^metric_par.mask_xo;     % masking across orientations
            k_mask_xn = 10.^metric_par.mask_xn;     % masking across neighboring bands

            self_mask = mutual_masking( b, o );            
            
            mask_xn = zeros( size( self_mask ) );
            if( b > 1 )
                mask_xn = max( imresize( mutual_masking( b-1, o ), size( self_mask ) ), 0 )/(band_norm/2);
            end
            if( b < (b_count-1) )
                mask_xn = mask_xn + max( imresize( mutual_masking( b+1, o ), size( self_mask ) ), 0 )/(band_norm*2);
            end
                        
            % correct activity for this particular channel
            band_mask_xo = max( mask_xo - self_mask, 0 );
            
            N_mask = band_norm * (k_mask_self*(self_mask./N_nCSF/band_norm).^q + ...
                k_mask_xo*(band_mask_xo./N_nCSF/band_norm).^q + ...
                k_mask_xn*(mask_xn./N_nCSF).^q);
            
            
            D = ex_diff./sqrt( N_nCSF.^(2*p) + N_mask.^2 );
%            if( b == b_count )
%                D_bands = set_band( D_bands, b, o, D );
%            else
                D_bands = set_band( D_bands, b, o, sign_pow(D/band_norm, pf)*band_norm );
%            end
        else
            % NO masking
            D = ex_diff./N_nCSF.^p;
            D_bands = set_band( D_bands, b, o, sign_pow(D/band_norm, pf)*band_norm );
        end
        
        % Quality prediction
        % test
        w_f = interp1( metric_par.quality_band_freq, metric_par.quality_band_w, ...
            clamp( band_freq(b), metric_par.quality_band_freq(end), metric_par.quality_band_freq(1) ) );
        epsilon = 1e-12;
        
        % Mask the pixels that are almost identical in test and
        % reference images. Improves predictions for small localized
        % differences.
        diff_mask_b = imresize( double(diff_mask), size( D ) );
        D_p = D .* diff_mask_b;
        
        Q = Q + (log( msre( D_p ) + epsilon )-log(epsilon)) * w_f;
        
        if( metric_par.do_quality_raw_data )
            qres = quality_pred_band( qres, D_p, b, o );
        end
        
    end
    
end

S_map = abs(reconSpyr( D_bands.pyr, D_bands.pind, metric_par.steerpyr_filter ));

% TODO: localized distortions may cause prediction of visibilble differences
% in other parts of an image because they affect low frequencies. This is
% especially apparent for super-threshold differences. A mechanism to
% restrict location of such changes is needed.
%
%S_map = S_map .* double(diff_mask);

if( metric_par.do_spatial_pooling )
    S_map = sum(S_map(:))/(max(S_map(:))+1e-12)*S_map;
end

res.P_map = 1 - exp( log(0.5)*S_map );
res.P_det = max( res.P_map(:) );

res.C_map = S_map;
res.C_max = max( S_map(:) );
res.Q = 100-Q;

% This MOS fitting did not work very well, disabled for now
%res.Q_MOS = 100 / (1+exp(metric_par.quality_logistic_q1*(Q+metric_par.quality_logistic_q2)));

if( metric_par.do_quality_raw_data )
    res.qres = qres;
end

    function m = mutual_masking( b, o )
        m = min( abs(get_band(B_T,b,o)), abs(get_band(B_R,b,o)) );
        % simplistic phase-uncertainty mechanism 
        % TODO - improve it
        
        if( metric_par.do_si_gauss ) 
            m = blur_gaussian( m, 10^metric_par.si_size );
        
        else
            F = ones( 3, 3 );
            m = conv2( m, F/numel(F), 'same');
        end
    end

end

function y = sign_pow( x, e )
y = sign(x) .* abs(x).^e;
end

function B = get_band( bands, b, o )

oc = min( o, bands.sz(b) );

B = pyrBand( bands.pyr, bands.pind, sum(bands.sz(1:(b-1)))+oc );

end

function sz = get_band_size( bands, b, o )
sz = bands.pind(sum(bands.sz(1:(b-1)))+o,:);
end

function bands = set_band( bands, b, o, B )

bands.pyr(pyrBandIndices(bands.pind,sum(bands.sz(1:(b-1)))+o)) = B;

end

function d = msre( X )

d = sqrt(sum(X(:).^2))/numel(X);

end

function L = display_model( V, gamma, peak, black_level )
L = peak * V.^gamma + black_level;
end

function RGB = display_model_srgb( sRGB )
a = 0.055;
thr = 0.04045;

RGB = zeros(size(sRGB));
RGB(sRGB<=thr) = sRGB(sRGB<=thr)/12.92;
RGB(sRGB>thr) = ((sRGB(sRGB>thr)+a)/(1+a)).^2.4;

RGB = 99*RGB + 1;

end

function RGB = xyz2rgb( XYZ )
% Transform image color space using matrix M
% dest = M * src

M = [ 3.240708 -1.537259 -0.498570;
    -0.969257  1.875995  0.041555;
    0.055636 -0.203996  1.057069 ];

RGB = reshape( (M * reshape( XYZ, [size(XYZ,1)*size(XYZ,2) 3] )')', ...
    [size(XYZ,1) size(XYZ,2) 3] );
end


function Y = spatial_summation( X, sigma )
% Essentilally a non-normalized Gaussian filter
% 

ksize = round(sigma*6);
h = fspecial( 'gaussian', ksize, sigma );
h = h / max(h(:));
Y = imfilter( X, h, 'replicate' );
    
end

function check_if_values_plausible( img, metric_par )
% Check if the image is in plausible range and report a warning if not.
% This is because the metric is often misused and used for with
% non-absolute luminace data.

if( ~metric_par.disable_lowvals_warning )
    if( max(img(:)) <= 1 ) 
        warning( 'hdrvdp:lowvals', [ 'The images contain very low physical values, below 1 cd/m^2. ' ...
            'The passed values are most probably not scaled in absolute units, as requied for this color encoding. ' ...
            'See ''doc hdrvdp'' for details. To disable this wanrning message, add option { ''disable_lowvals_warning'', ''true'' }.' ] );
    end
end

end

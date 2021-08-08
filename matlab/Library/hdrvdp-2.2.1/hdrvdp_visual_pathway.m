function [bands, L_adapt, band_freq, bb_padvalue] = hdrvdp_visual_pathway( img, name, metric_par, bb_padvalue )
% HDRVDP_VISUAL_PATHWAY (internal) Process image along the visual pathway
% to compute normalized perceptual response
%
% img - image data (can be multi-spectral)
% name - string with the name of this map (shown in warnings and error
%        messages)
% options - cell array with the 'option', value pairs
% bands - CSF normalized freq. bands
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

global hdrvdp_cache;

if( any( isnan( img(:) ) ) )
    warning( 'hdrvdp:BadImageData', '%s image contains NaN pixel values', name );
    img(isnan(img)) = 1e-5;
end

% =================================
% Precompute
%
% precompute common variables and structures or take them from cache
% =================================

width = size(img,2);
height = size(img,1);
img_sz = [height width]; % image size
img_ch = size(img,3); % number of color channels

rho2 = create_cycdeg_image( img_sz*2, metric_par.pix_per_deg ); % spatial frequency for each FFT coefficient, for 2x image size

if( metric_par.do_mtf )
    mtf_filter = hdrvdp_mtf( rho2, metric_par ); % only the last param is adjusted
else
    mtf_filter = ones( size(rho2) );
end

if( ~metric_par.debug )
    % memory savings for huge images
    clear rho2;
end

% Load spectral sensitivity curves
[lambda, LMSR_S] = load_spectral_resp( 'log_cone_smith_pokorny_1975.csv' );
LMSR_S(LMSR_S==0) = min(LMSR_S(:));
LMSR_S = 10.^LMSR_S;

[~, ROD_S] = load_spectral_resp( 'cie_scotopic_lum.txt' );
LMSR_S(:,4) = ROD_S;

IMG_E = metric_par.spectral_emission;

% =================================
% Precompute photoreceptor non-linearity
% =================================

pn = hdrvdp_get_from_cache( 'pn', [metric_par.rod_sensitivity metric_par.csf_sa], @() create_pn_jnd( metric_par ) );

pn.jnd{1} = pn.jnd{1} * metric_par.sensitivity_correction;
pn.jnd{2} = pn.jnd{2} * metric_par.sensitivity_correction;


% =================================
% Optical Transfer Function
% =================================

L_O = zeros( size(img) );
for k=1:img_ch  % for each color channel
    if( metric_par.do_mtf )
        % Use per-channel or one per all channels surround value
        pad_value = metric_par.surround_l( k );
        L_O(:,:,k) =  clamp( fast_conv_fft( double(img(:,:,k)), mtf_filter, pad_value ), 1e-5, 1e10 );
    else
        % NO mtf
        L_O(:,:,k) =  img(:,:,k);
    end    
end

if( ~metric_par.debug )
    % memory savings for huge images
    clear mtf_filter;
end

%TODO - MTF reduces luminance values

% =================================
% Photoreceptor spectral sensitivity
% =================================

M_img_lmsr = zeros( img_ch, 4 ); % Color space transformation matrix
for k=1:4
    for l=1:img_ch
        M_img_lmsr(l,k) = trapz( lambda, LMSR_S(:,k).*IMG_E(:,l) );                
    end
end

% Color space conversion
R_LMSR = reshape( reshape( L_O, width*height, img_ch )*M_img_lmsr, height, width, 4 );

if( ~metric_par.debug )
    % memory savings for huge images
    clear L_O;
end


%surround_LMSR = metric_par.surround_l * M_img_lmsr;

% =================================
% Adapting luminance
% =================================

L_adapt = R_LMSR(:,:,1) + R_LMSR(:,:,2);

% =================================
% Photoreceptor non-linearity
% =================================

%La = mean( L_adapt(:) );

P_LMR = zeros(height, width, 4);
%surround_P_LMR = zeros(1,4);
for k=[1:2 4] % ignore S - does not influence luminance   
    if( k==4 )
        ph_type = 2; % rod
        ii = 3;
    else
        ph_type = 1; % cone
        ii = k;
    end
    
    P_LMR(:,:,ii) = pointOp( log10( clamp(R_LMSR(:,:,k), 10^pn.Y{ph_type}(1), 10^pn.Y{ph_type}(end)) ), ...
        pn.jnd{ph_type}, pn.Y{ph_type}(1), pn.Y{ph_type}(2)-pn.Y{ph_type}(1), 0 );
    
%    surround_P_LMR(ii) = interp1( pn_Y{ph_type}, pn_jnd{ph_type}, ...
%        log10( clamp(surround_LMSR(k), 10^pn_Y{ph_type}(1), 10^pn_Y{ph_type}(end)) ) );
end

if( ~metric_par.debug )
    % memory savings for huge images
    clear R_LMSR;
end


% =================================
% Remove the DC component, from 
% cone and rod pathways separately
% =================================

% TODO - check if there is a better way to do it
% cones
P_C = P_LMR(:,:,1)+P_LMR(:,:,2);
mm = mean(mean( P_C ));
P_C = P_C - mm;
% rods
mm = mean(mean( P_LMR(:,:,3) ));
P_R = P_LMR(:,:,3) - mm;


% =================================
% Achromatic response
% =================================

P = P_C + P_R;

if( ~metric_par.debug )
    % memory savings for huge images
    clear P_LMR P_C P_R;
end


%surround_P = surround_P_LMR(1)+surround_P_LMR(2)+surround_P_LMR(3);

% =================================
% Multi-channel decomposition
% =================================


%[lo0filt,hi0filt,lofilt,bfilts,steermtx,harmonics] = eval(metric_par.steerpyr_filter);
%max_ht = maxPyrHt(size(P), size(lofilt,1));
%[bands.pyr,bands.pind] = buildSpyr( P, min(max_ht,6), metric_par.steerpyr_filter );

[bands.pyr,bands.pind] = buildSpyr( P, 'auto', metric_par.steerpyr_filter );

bands.sz = ones( spyrHt( bands.pind ) + 2, 1 );
bands.sz(2:end-1) = spyrNumBands( bands.pind );

band_freq = 2.^-(0:(spyrHt( bands.pind )+1)) * metric_par.pix_per_deg / 2;

% CSF-filter the base band
L_mean = mean( L_adapt(:) );
    
BB = pyrBand( bands.pyr, bands.pind, sum(bands.sz) );
% I wish to know why sqrt(2) works better, as it should be 2
rho_bb = create_cycdeg_image( size(BB)*2, band_freq(end)*2*sqrt(2) ); 
csf_bb = hdrvdp_ncsf( rho_bb, L_mean, metric_par );
    
% pad value must be the same for test and reference images. 
if( bb_padvalue == -1 )
    bb_padvalue = mean(BB(:));
end
    
bands.pyr(pyrBandIndices(bands.pind,sum(bands.sz))) = fast_conv_fft( BB, csf_bb, bb_padvalue );


if( 0 )
for b=1:length(bands.sz)
    for o=1:bands.sz(b)
   
        b_ind = sum(bands.sz(1:(b-1)))+o;
        BB = pyrBand( bands.pyr, bands.pind, b_ind );
        rho_bb = create_cycdeg_image( size(BB)*2, band_freq(b)*4 );
        csf_bb = hdrvdp_csf( rho_bb, L_mean, metric_par );
    
        if( metric_par.surround_l == -1 )
            pad_value = mean(BB(:));
        else
            pad_value = metric_par.surround_l;
        end
    
        bands.pyr(pyrBandIndices(bands.pind,b_ind)) = fast_conv_fft( BB, csf_bb, pad_value );

    end
    
end
end


    function item = cache_get( item_name, item_signature, compute_func )
        sign_name = [ item_name '_sign' ];
        if( isfield( hdrvdp_cache, sign_name ) && all( hdrvdp_cache.( sign_name ) == item_signature) )  % caching
            item = hdrvdp_cache.( item_name );
        else
            item = compute_func();
            hdrvdp_cache.( sign_name ) = zeros(size(item_signature)); % in case of breaking at this point
            hdrvdp_cache.( item_name ) = item;
            hdrvdp_cache.( sign_name ) = item_signature;
        end
    end

end


function [Y jnd] = build_jndspace_from_S(l,S)

L = 10.^l;
dL = zeros(size(L));

for k=1:length(L)
    thr = L(k)/S(k);

    % Different than in the paper because integration is done in the log
    % domain - requires substitution with a Jacobian determinant
    dL(k) = 1/thr * L(k) * log(10);
end

Y = l;
jnd = cumtrapz( l, dL );

end

function pn = create_pn_jnd( metric_par )
% Create lookup tables for intensity -> JND mapping

c_l = logspace( -5, 5, 2048 );

s_A = hdrvdp_joint_rod_cone_sens( c_l, metric_par );
s_R = hdrvdp_rod_sens( c_l, metric_par ) * 10.^metric_par.rod_sensitivity;

% s_C = s_L = S_M
s_C = 0.5 * interp1( c_l, max(s_A-s_R, 1e-3), min( c_l*2, c_l(end) ) );

pn = struct();

[pn.Y{1} pn.jnd{1}] = build_jndspace_from_S( log10(c_l), s_C );
[pn.Y{2} pn.jnd{2}] = build_jndspace_from_S( log10(c_l), s_R );

end

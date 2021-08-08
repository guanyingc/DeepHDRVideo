%% HDR-VDP-2: A calibrated visual metric for visibility and quality predictions in all luminance conditions
% matlab version
%
% -----------------------------------------------------------------
% Documentation
%
% HDRVDP - run HDR-VDP-2 on a pair of images
% HDRVDP_VISUALIZE - produce visualization of the probability maps
% HDRVDP_PIX_PER_DEG - compute an angular resolution required for HDR-VDP-2
%
% -----------------------------------------------------------------
% Example: 
%  
%  % Load reference and test images
%  T = double(imread( 'DistortedImage.png' ))/2^8; % images must be
%                                                  % normalized 0-1
%  R = double(imread( 'OriginalImage.png' ))/2^8;
%
%  % Compute pixels per degree for the viewing conditions
%  ppd = hdrvdp_pix_per_deg( 21, [size(O,2) size(O,1)], 1 );
%
%  % Run hdrvdp
%  res1 = hdrvdp( T, R, 'sRGB-display', ppd )
%
% Copyright (c) 2004-2011, Rafal Mantiuk <mantiuk@gmail.com>
%
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

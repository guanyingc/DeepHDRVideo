function ppd = hdrvdp_pix_per_deg( display_diagonal_in, resolution, viewing_distance )
% HDRVDP_PIX_PER_DEG - computer pixels per degree given display parameters
% and viewing distance
%
% ppd = hdrvdp_pix_per_deg( display_diagonal_in, resolution,
% viewing_distance )
%
% This is a convenience function that can be used to provide angular
% resolution of input images for the HDR-VDP-2. 
%
% display_diagonal_in - diagonal display size in inches, e.g. 19, 14
% resolution - display resolution in pixels as a vector, e.g. [1024 768]
% viewing_distance - viewing distance in meters, e.g. 0.5
%
% Note that the function assumes 'square' pixels, so that the aspect ratio
% is resolution(1):resolution(2).
%
% EXAMPLE:
% ppd = hdrvdp_pix_per_deg( 24, [1920 1200], 0.5 );
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

ar = resolution(1)/resolution(2);

height_mm = sqrt( (display_diagonal_in*25.4)^2 / (1+ar^2) );

height_deg = 2 * atand( 0.5*height_mm/(viewing_distance*1000) );

ppd = resolution(2)/height_deg;

end
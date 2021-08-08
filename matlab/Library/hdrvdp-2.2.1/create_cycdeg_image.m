function D = create_cycdeg_image( im_size, pix_per_deg )
% CREATE_CYCDEG_IMAGE (internal) create matrix that contains frequencies,
% given in cycles per degree.
%
% D = create_cycdeg_image( im_size, pix_per_deg )
% im_size     - [height width] vector with image size
% pix_per_deg - pixels per degree for both horizontal and vertical axis
%               (assumes square pixels)
%
% Useful for constructing Fourier-domain filters based on OTF or CSF data.
%
% (C) Rafal Mantiuk <mantiuk@gmail.com>
% This is an experimental code for internal use. Do not redistribute.

nyquist_freq = 0.5 * pix_per_deg;
half_size = floor(im_size/2);
odd = mod( im_size, 2 );
freq_step = nyquist_freq./half_size;

if( odd(2) )
    xx = [ linspace( 0, nyquist_freq, half_size(2)+1 ) linspace( -nyquist_freq, -freq_step(2), half_size(2) ) ];
else
    xx = [ linspace( 0, nyquist_freq-freq_step(2), half_size(2) ) linspace( -nyquist_freq, -freq_step(2), half_size(2) ) ];
end

if( odd(1) )
    yy = [ linspace( 0, nyquist_freq, half_size(1)+1 ) linspace( -nyquist_freq, -freq_step(1), half_size(1) ) ];
else
    yy = [ linspace( 0, nyquist_freq-freq_step(1), half_size(1) ) linspace( -nyquist_freq, -freq_step(1), half_size(1) ) ];
end

[XX YY] = meshgrid( xx, yy );

D = sqrt( XX.^2 + YY.^2 );

%[XX YY] = meshgrid( linspace( 0, nyquist_freq, half_size(2)+even(2) ), linspace( 0, nyquist_freq, half_size(1)+even(1) ) );
%D1 = sqrt( XX.^2 + YY.^2 );
%[XX YY] = meshgrid( linspace( nyquist_freq-frec_step(2), frec_step(2), half_size(2) ), linspace( 0, nyquist_freq, half_size(1)+even(1) ) );
%D2 = sqrt( XX.^2 + YY.^2 );
%[XX YY] = meshgrid( linspace( 0, nyquist_freq, half_size(2)+even(2) ), linspace( nyquist_freq-frec_step(1), frec_step(1), half_size(1) ) );
%D3 = sqrt( XX.^2 + YY.^2 );
%[XX YY] = meshgrid( linspace( nyquist_freq-frec_step(2), frec_step(2), half_size(2) ), linspace( nyquist_freq-frec_step(1), frec_step(1), half_size(1) ) );
%D4 = sqrt( XX.^2 + YY.^2 );
%D = [ D1 D2; D3 D4 ];

%[XX YY] = meshgrid( linspace( 0, nyquist_freq, half_size(2)+even(2) ), linspace( 0, nyquist_freq, half_size(1)+even(1) ) );
%D1 = sqrt( XX.^2 + YY.^2 );
%[XX YY] = meshgrid( linspace( nyquist_freq-frec_step(2), frec_step(2), half_size(2) ), linspace( 0, nyquist_freq, half_size(1)+even(1) ) );
%D2 = sqrt( XX.^2 + YY.^2 );
%[XX YY] = meshgrid( linspace( 0, nyquist_freq, half_size(2)+even(2) ), linspace( nyquist_freq-frec_step(1), frec_step(1), half_size(1) ) );
%D3 = sqrt( XX.^2 + YY.^2 );
%[XX YY] = meshgrid( linspace( nyquist_freq-frec_step(2), frec_step(2), half_size(2) ), linspace( nyquist_freq-frec_step(1), frec_step(1), half_size(1) ) );
%D4 = sqrt( XX.^2 + YY.^2 );

end
function Y = fast_gauss( X, sigma, do_norm )
% Low-pass filter image using the Gaussian filter
% 
% Y = blur_gaussian( X, sigma, do_norm )
%  
% do_norm - normalize the results or not, 'true' by default
%
% Use FFT or spatial domain, whichever is faster

%ksize2 = min(max(3, ceil(((sigma-0.8)/0.3 + 1)*2)), max(size(X))*2);

if( ~exist( 'do_norm', 'var' ) )
    do_norm = true;
end

if( do_norm )
    norm_f = 1;
else
    norm_f = sigma*sqrt(2*pi);
end


if( sigma >= 4.3 ) % Experimentally found threshold when FFT is faster
   
    ks = [size(X,1) size(X,2)]*2;
       
    NF = pi;
%    xx = [linspace( 0, NF, ks(2)/2 ) linspace(-NF, 0, ks(2)/2) ];
%    yy = [linspace( 0, NF, ks(1)/2 ) linspace(-NF, 0, ks(1)/2) ];
    xx = [linspace( 0, NF, ks(2)/2 ) linspace(-NF, -NF/(ks(2)/2), ks(2)/2) ];
    yy = [linspace( 0, NF, ks(1)/2 ) linspace(-NF, -NF/(ks(1)/2), ks(1)/2) ];
    
    [XX YY] = meshgrid( xx, yy );
    
    K = exp( -0.5*(XX.^2 + YY.^2)*sigma^2 ) * norm_f;
    
    Y = zeros( size(X) );
    for cc=1:size(X,3)
        Y(:,:,cc) = fast_conv_fft( X(:,:,cc), K, 'replicate' );
    end        
    
else

    ksize = round(sigma*5);   
    h = fspecial( 'gaussian', ksize, sigma ) * norm_f;
    Y = imfilter( X, h, 'replicate' );    
    
end

end


function Y = fast_conv_fft( X, fH, pad_value )
% Convolve with a large support kernel in the Fourier domain.
%
% Y = fast_conv_fft( X, fH, pad_value )
%
% X - image to be convolved (in spatial domain)
% fH - filter to convolve with in the Fourier domain, idealy 2x size of X
% pad_value - value to use for padding when expanding X to the size of fH
%
% (C) Rafal Mantiuk <mantiuk@gmail.com>
% This is an experimental code for internal use. Do not redistribute.

pad_size = (size(fH)-size(X));

%mX = mean( X(:) );

fX = fft2( padarray( X, pad_size, pad_value, 'post' ) );

Yl = real(ifft2( fX.*fH, size(fX,1), size(fX,2), 'symmetric' ));

Y = Yl(1:size(X,1),1:size(X,2));

end

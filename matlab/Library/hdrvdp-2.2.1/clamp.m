function Y = clamp( X, min, max )
% CLAMP restricts values of 'X' to be within the range from 'min' to 'max'.
%
% Y = clamp( X, min, max )
%  
% (C) Rafal Mantiuk <mantiuk@gmail.com>
% This is an experimental code for internal use. Do not redistribute.

  Y = X;
  Y(X<min) = min;
  Y(X>max) = max;
end

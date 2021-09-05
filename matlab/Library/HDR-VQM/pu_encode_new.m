function luma = pu_encode_new(luminance)
load pu_space.mat
% Convert luminance to perceptually uniform code values that are
% backward-compatible with sRGB. Refer to [1] for details. Please cite 
% the paper if you find our work useful. The paper and bib file can be
% found at the project website: 
% http://www.mpi-inf.mpg.de/resources/hdr/fulldr_extension/
%
% [1] T. O. Ayd{\i}n, R. Mantiuk, and H.-P. Seidel, "Extending quality 
% metrics to full dynamic range images," in Human Vision and Electronic "
% Imaging XIII, Proceedings of SPIE, (San Jose, USA), January 2008
% 
% Usage: 
% pu_space = read_csv('pu_space.csv');
% luma = pu_encode([1 10; 100 1000], pu_space)

luma = interp1(log10(pu_space(:,1)), log10(pu_space(:,2)), ...
    log10(luminance));
% luma = interp1((pu_space(:,1)), (pu_space(:,2)), ...
%      abs(luminance));
luma = 10.^luma;

luma=RemoveSpecials(luma);
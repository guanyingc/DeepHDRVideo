function subband_errors = subband_errors(I,I_perturbed,n_scale,n_orient) 
%assuming full HDR frames, downscale to be compatible with block size
r = 512;
c = 896;
I = imresize(I,[r c]);
I_perturbed = imresize(I_perturbed,[r c]);
[row,col] = size(I);
sum_matrix = zeros(row,col);
%Gabor decomposition with 5 scales and 4 orientation levels
weight_csf = [0.6076    1.0153    1.3283    1.5587    1.7188];
p = gabor_hdrvqm(I,n_scale,n_orient);
p_perturbed = gabor_hdrvqm(I_perturbed,n_scale,n_orient);
for m =1:n_scale
        for n =1:n_orient
		    %employ only magnitude of convolution for each scale and orientation level
			%ignore phase in this version
			mag_ref = abs(p{m,n});
			mag_ref = RemoveSpecials(mag_ref);
			mag_dis = abs(p_perturbed{m,n});
			mag_dis = RemoveSpecials(mag_dis);
			efm{m,n} = similarity_error_band(mag_ref ,mag_dis);
			distance(n) = mean2(efm{m,n});
			sum_matrix = sum_matrix + efm{m,n}; %simple summation/mean, equal weighting to all scale and orientation levels
			clear mag_ref mag_dis
        end
		%distance
		%[value,index] = min(distance);
		%sprintf('maximally distirted orientation is %d',index)
		%sum_matrix = sum_matrix + efm{m,index}; %summation across maximally disoirted orientation
		%clear distance
end
subband_errors = sum_matrix; 

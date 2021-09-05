function imgout = clip_luminance(frame,format,cfg_hdrvqm)

%define the black point, lowest displayable output
%Our measurement probe precision is about 0.1 cd/m2 
lowest_lum_display = cfg_hdrvqm.min_display; 
%maximum displayable value when sufficiently big area is bright
%Will be less than theoretical maximum
max_lum_display = cfg_hdrvqm.max_display;     
frame = double(frame);
lum_frame = 0.2126*frame(:,:,1) + 0.7152*frame(:,:,2) + 0.0722*frame(:,:,3);

switch format
		case ('.hdr')
			switch cfg_hdrvqm.do_adapt
				case('none')
					lum_frame_emitted = lum_frame; %no scaling
				case('linear')
					% Use luminous efficacy value for RGBE format
					lum_frame_emitted = lum_frame*179;
				case('DM')
					lum_frame_emitted = lum_frame; %no scaling	
		end
		case ('.exr')
		lum_frame_emitted = lum_frame;
end
%apply clipping
lum_frame_emitted(lum_frame_emitted < lowest_lum_display) = lowest_lum_display;
lum_frame_emitted(lum_frame_emitted > max_lum_display) = max_lum_display;


imgout = zeros(size(frame));
for i=1:3
    imgout(:,:,i) = (frame(:,:,i).*lum_frame_emitted)./lum_frame;
end
imgout = RemoveSpecials(imgout);

function error_frame = hdrvqm_perframe_error(frame_count,path_src_emitted, path_hrc_emitted, cfg_hdrvqm)
%config_hdrvqm
%global cfg_hdrvqm;
Images_List_HDR_src = dir(fullfile(path_src_emitted, '*.hdr'));
Images_List_EXR_src = dir(fullfile(path_src_emitted, '*.exr'));
Images_List_HDR_hrc = dir(fullfile(path_hrc_emitted, '*.hdr'));
Images_List_EXR_hrc = dir(fullfile(path_hrc_emitted, '*.exr'));
Images_List_src = [Images_List_HDR_src;Images_List_EXR_src];
Images_List_hrc = [Images_List_HDR_hrc;Images_List_EXR_hrc];
%fprintf('Reference/Distorted file number: %d/%d\n', length(Images_List_src), length(Images_List_hrc))
if (length(Images_List_src) < 1)
    error('No groundtruth HDRs found!')
end
if (length(Images_List_hrc) < 1)
    error('No estimated HDRs found!')
end
if (length(Images_List_src) ~= length(Images_List_hrc))
    error('Length of groundtruth does not match the estimated HDRs!')
end

for index = 1:length(Images_List_src)
    Image_Name_src{index} = Images_List_src(index).name;
end
for index = 1:length(Images_List_hrc)
    Image_Name_hrc{index} = Images_List_hrc(index).name;
end
%for index = 1:length(Image_Name_hrc)
%    fprintf('%s %s\n', Image_Name_src{index}, Image_Name_hrc{index})
%end

flag_format = find([length(Images_List_HDR_src) length(Images_List_EXR_src)]);
if(flag_format ==1)
    format = '.hdr';
else
    format = '.exr';
end
fprintf('%s\n', fullfile(path_src_emitted, Image_Name_src{frame_count}), fullfile(path_hrc_emitted, Image_Name_hrc{frame_count}))
if(strcmp(Image_Name_src{frame_count}(end-3:end),'.hdr'))
    src_value = (hdrimread(fullfile(path_src_emitted, Image_Name_src{frame_count}))*4500);
else
    src_value = exrread(fullfile(path_src_emitted, Image_Name_src{frame_count}));
end
src_value = clip_luminance(src_value,format,cfg_hdrvqm);
org_frame(frame_count).name = 'cdata';
I_org = (double(lum(src_value)));
org_frame(frame_count).cdata = RemoveSpecials(I_org);
clear original_value I_org


if(strcmp(Image_Name_hrc{frame_count}(end-3:end),'.hdr'))
    hrc_value = (hdrimread(fullfile(path_hrc_emitted, Image_Name_hrc{frame_count}))*4500/cfg_hdrvqm.est_hdr_max);
else
    hrc_value = exrread(fullfile(path_hrc_emitted, Image_Name_hrc{frame_count}));
end
hrc_value = clip_luminance(hrc_value,format,cfg_hdrvqm);
dis_frame(frame_count).name = 'cdata';
I_dis = (double(lum(hrc_value)));
dis_frame(frame_count).cdata = RemoveSpecials(I_dis);
clear distorted_value I_dis


error_frame = subband_errors(((((double(pu_encode_new((org_frame(frame_count).cdata))))))),(((((double(pu_encode_new((dis_frame(frame_count).cdata)))))))),cfg_hdrvqm.n_scale,cfg_hdrvqm.n_orient);
end

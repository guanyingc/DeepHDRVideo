function error_frame = hdrvqm_perframe_error2(src_hdr_path, est_hdr_path, cfg_hdrvqm)
%config_hdrvqm
%global cfg_hdrvqm;
%fprintf('%s %s\n', src_hdr_path, est_hdr_path);

if(strcmp(src_hdr_path(end-3:end),'.hdr'))
    src_value = (hdrimread(src_hdr_path)*4500);
else
    src_value = exrread(src_hdr_path);
end

format = '.hdr';
src_value = clip_luminance(src_value,format,cfg_hdrvqm);
%org_frame(frame_count).name = 'cdata';
I_org = (double(lum(src_value)));
org_frame = RemoveSpecials(I_org);
clear original_value I_org

if(strcmp(est_hdr_path(end-3:end),'.hdr'))
    hrc_value = (hdrimread(est_hdr_path) .* 4500 ./ cfg_hdrvqm.est_hdr_max);
else
    hrc_value = exrread(est_hdr_path);
end

hrc_value = clip_luminance(hrc_value,format,cfg_hdrvqm);
%dis_frame(frame_count).name = 'cdata';
I_dis = (double(lum(hrc_value)));
dis_frame = RemoveSpecials(I_dis);
clear distorted_value I_dis

error_frame = subband_errors(double(pu_encode_new(org_frame)),double(pu_encode_new(dis_frame)),cfg_hdrvqm.n_scale, cfg_hdrvqm.n_orient);
end

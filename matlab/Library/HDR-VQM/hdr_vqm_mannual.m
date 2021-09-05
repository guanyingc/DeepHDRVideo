function HDRVQM = hdr_vqm_mannual(src,hrc,frame_count,total_frame)
%set the default parameters, run the config file
persistent error_video_hdrvqm
%config_hdrvqm
global cfg_hdrvqm;

error_video_hdrvqm(:,:,frame_count) = subband_errors(double(pu_encode_new((src))),double(pu_encode_new(hrc)),cfg_hdrvqm.n_scale,cfg_hdrvqm.n_orient);
if(frame_count==total_frame)
    switch cfg_hdrvqm.data
    case('image')
        HDRVQM = st_pool(st_pool(error_video_hdrvqm,0.5),0.5);
        fprintf('\nHDR-VQM for image is: %f\n',HDRVQM)
        clear error_video_hdrvqm
        matlabpool close force local
    case('video')
            HDRVQM = hdrvqm_error_pooling(error_video_hdrvqm,cfg_hdrvqm);
            fprintf('\nHDR-VQM for video is: %f\n',HDRVQM)
            clear error_video_hdrvqm
            matlabpool close force local
        end	
    else
            HDRVQM=-1;
        end
end




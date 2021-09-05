function res = hdrvqm_error_pooling(video,cfg_hdrvqm)
[row,col,length_video] = size(video);
%specify block size
indx = 1;
%generate non-overlapping chunks of error video
for count = 1:cfg_hdrvqm.n_frames_fixate:length_video-cfg_hdrvqm.n_frames_fixate
    clear yref 
    n = 1;
    for j = count:count + cfg_hdrvqm.n_frames_fixate-1
        yref(:,:,n) = video(:,:,j);
        n = n + 1;
    end
    ST_v_ts(:,:,indx) = hdrvqm_short_term_temporal_pooling(yref,cfg_hdrvqm);
    indx = indx + 1;
end
[r,c,t] = size(ST_v_ts);
ST_v_ts = reshape(ST_v_ts,r*c,t);
res = st_pool(st_pool(ST_v_ts,cfg_hdrvqm.perc),cfg_hdrvqm.perc);





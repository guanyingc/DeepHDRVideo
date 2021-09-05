function res = hdrvqm_short_term_temporal_pooling_own(video,cfg_hdrvqm)
cfg_hdrvqm.bsize = ceil(cfg_hdrvqm.bsize);
[row,col,length_video] = size(video);
% video = video.cdata;
%pad zeros, if required
for frame_count = 1:length_video
    [r,c]=size(video(:,:,1));
    zr = ceil(cfg_hdrvqm.bsize*ceil(r/cfg_hdrvqm.bsize)-r);
    zc= ceil(cfg_hdrvqm.bsize*ceil(c/cfg_hdrvqm.bsize)-c);
    resized_video{frame_count} = padarray(video(:,:,frame_count),[zr zc],'post');
end

%split into non-overlapping blocks
for frame_count = 1:length_video 
    [p1,p2]=size(resized_video{frame_count});
    I = resized_video{frame_count}; %current frame
    I_cell{frame_count}=mat2cell(I,[cfg_hdrvqm.bsize*ones(1,p1/cfg_hdrvqm.bsize)],[cfg_hdrvqm.bsize*ones(1,p2/cfg_hdrvqm.bsize)]);
end

%compute statistics (eg. standard deviation) for each
%ST tube made from each block per error frame
for frame_count = 1:length_video

    [p,q]=size(I_cell{frame_count});
    %
    for i=1:(p*q)
        er = (I_cell{frame_count}); %each frame 
        blck{i,frame_count} = er{i};
    end
end

for i=1:(p*q)   %process for each block
    g = 1;
    for frame_count = 1:length_video   %process for each frame
        tmp = ((blck{i,frame_count})); %ith block for all frames
        fg(:,g) = tmp(:);
        clear tmp
        g = g + 1;
    end
    %compute deviation/variance for ith ST tube
    res(i) = std2(fg);
    %res(i) = mean2(fg)/(std2(fg)+.8);
    clear fg
end
res = reshape(res,p,q);
end

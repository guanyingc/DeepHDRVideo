function main_eval(nexps, method)

addpath(genpath('Library'));

config_eval(nexps, method)
global conf;

config_hdrvqm;
global cfg_hdrvqm;

ndataset = length(conf.datasets);

for id = 1: ndataset
    data_name = conf.datasets{id};
    summary_filename = get_summary_filename('results', data_name);

    gt_dir = conf.gt.(data_name).path;
    est_dir = conf.est.(data_name).path;
    est_hdr_max = conf.est.(data_name).est_hdr_max;
    expos = get_expo_types(gt_dir);

    max_num = conf.max_num;
    if max_num <= 0
        max_num = length(expos);
    else
        max_num = min(max_num, length(expos));
    end

    fprintf('***** GT dir: %s, %d Expo , Max num: %d\n', gt_dir, length(expos), max_num)
    fprintf('***** Est dir: %s***** \n', est_dir)

    [psnrTs, psnrLs, ~, hdrvdps] = paral_eval_HDRs(gt_dir, est_dir, max_num, est_hdr_max); % parfor loop
    all_results = cat(2, psnrTs, psnrLs, hdrvdps);

    num = size(all_results, 1) - 1;
    metric_name = sprintf('Data_%s_%d_full.txt', date, num);
    dlmwrite(fullfile(est_dir, metric_name), all_results, 'precision', '%.6f', 'delimiter', ' ');

    % Save results
    fprintf('****Wrinting data to files*****\n')
    [summary_str, tex_summary_str] = save_hdr_txt_results(est_dir, all_results, expos(1:num), summary_filename);

    fprintf('****Wrinting summary data to %s*****\n', summary_filename)
    fid = fopen(summary_filename, 'a');
    fwrite(fid, sprintf('\n[%s %dExpo] est_dir: %s\n', method, nexps, est_dir));

    %fwrite(fid, hdrvqm_str);

    for i = 1: length(summary_str)
        fprintf(summary_str{i});
    end
    for i = 1: length(tex_summary_str)
        fprintf(tex_summary_str{i});
    end
    %fprintf(hdrvqm_str);

    fclose(fid);
end


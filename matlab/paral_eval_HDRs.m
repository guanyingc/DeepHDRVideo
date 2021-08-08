function [psnrT_all, psnrL_all, ssimT_all, hdrvdp_all] = eval_HDRs2(gt_dir, est_dir, max_num, est_hdr_max)

gt_hdr_list = get_hdr_list(gt_dir);
est_hdr_list = get_hdr_list(est_dir);

gt_hdr_n = numel(gt_hdr_list);
est_hdr_n = numel(est_hdr_list);

if (gt_hdr_n < 1)
    error('No HDR founds in gt_dir!')
end
if (est_hdr_n < 1)
    error('No HDR founds in est_dir!')
end
if (gt_hdr_n ~= est_hdr_n)
    error('Length of GT (%d) does not equal to length of Estimation (%d)', gt_hdr_n, est_hdr_n)
end

if ~exist('max_num', 'var') | max_num < 0
    num = gt_hdr_n;
else
    num = max_num;
end

if ~exist('est_hdr_max', 'var')
    est_hdr_max = 1.;
end

fprintf('Found %d HDRs, test %d HDRs\n', gt_hdr_n, num)
psnrT_all = zeros(num, 1);
psnrL_all = zeros(num, 1);
ssimT_all = zeros(num, 1);
hdrvdp_all = zeros(num, 1);

delete(gcp('nocreate'))
p = parpool('local', 16);
parfor i = 1: num
    gt_hdr = double(hdrread(fullfile(gt_dir, gt_hdr_list{i})));
    est_hdr = double(hdrread(fullfile(est_dir, est_hdr_list{i}))) ./ est_hdr_max;

    % Compute PSNR in the linear domain
    psnrL_all(i) = psnr(est_hdr, gt_hdr);

    % Compute PSNR in the log domain
    log_gt = mulog_tonemap(gt_hdr, 5000);
    log_est = mulog_tonemap(est_hdr, 5000);
    psnrT_all(i) = psnr(log_est, log_gt);
    psnrT2 = my_psnr(log_est, log_gt);
    
    % Compute  HDR-VDP
    ppd = hdrvdp_pix_per_deg(24, [size(gt_hdr,2) size(gt_hdr, 1)], 0.5);
    hdrvdp_res = hdrvdp(est_hdr, gt_hdr, 'sRGB-display', ppd);
    hdrvdp_all(i) = hdrvdp_res.Q;

    % Compute SSIM
    %ssimT_all(i) = ssim(log_est, log_gt);
    %fprintf('sim\n')
    fprintf('%d/%d: PSNR-T %.2f, PSNR-L %.2f, SSIM %.3f HDR-VDP %.2f\n', i, num, psnrT_all(i), psnrL_all(i), ssimT_all(i), hdrvdp_all(i));
end
delete(p)

psnrT_all = [psnrT_all; mean(psnrT_all)];
psnrL_all = [psnrL_all; mean(psnrL_all)];
ssimT_all = [ssimT_all; mean(ssimT_all)];
hdrvdp_all = [hdrvdp_all; mean(hdrvdp_all)];

fprintf('[Average] PSRN: %f, PSNR-L: %.2f, SSIM: %.3f, HDR-VDP: %.2f\n', psnrT_all(end), psnrL_all(end), ssimT_all(end), hdrvdp_all(end));
end


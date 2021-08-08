function psnr = my_psnr(gt, est)
    diff = gt - est;
    square_diff = diff.^2;
    psnr = -10 * log10(mean(square_diff(:))+1e-8);
end

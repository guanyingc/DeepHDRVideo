T = hdrimread('demo/gt_hdrs/09-25-15.54.49_seq_grab_loop_0912_3stop_2exps_6s_wb.hdr');
R = hdrimread('demo/refine_est_hdrs/0000_09-25-15.54.49_seq_grab_loop_0912_3stop_2exps_6s_wb_img_07.hdr');
%R = hdrimread('demo/est_hdrs/0000_09-25-15.54.49_seq_grab_loop_0912_3stop_2exps_6s_wb_img_07.hdr');
ppd = hdrvdp_pix_per_deg(24, [size(T,2) size(T, 1)], 0.5);

res = hdrvdp(T, R, 'sRGB-display', ppd)

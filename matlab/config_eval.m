function config_eval(nexps, method)
global conf;

conf.max_num = -1;

% Setup GT path
if nexps == 2 % two exposures
    % dynamic dataset
    conf.gt.realD.path = '../data/dynamic_RGB_data_2exp_release/';
    % static dataset
    conf.gt.realS.path = '../data/static_RGB_data_2exp_rand_motion_release';

elseif nexps == 3 % three exposures
    conf.gt.realD.path = '../data/dynamic_RGB_data_2exp_release/';
    conf.gt.realS.path = '../data/static_RGB_data_3exp_rand_motion_release/';
end

%conf.datasets = {'realD', 'realS'};
conf.datasets = {'realS'};

for i = 1: length(conf.datasets)
    conf.est.(conf.datasets{i}).est_hdr_max = 1; % parameter to scale the HDR estimated HDR
end

if nexps == 2
    if strcmp(method, 'Ours')
        %conf.est.realD.path = '';
        conf.est.realS.path = '../data/models/CoarseToFine_2Exp/08-08,refine_net,weight_net,DAHDRnet,TA2,hdr2E_flow2s_model,real_benchmark_dataset,hd_w-1,cm_d-256,fa_k-8,s2_inexpm,cached_data,static_RGBon_release/test/Details/01/'
    else
        error('Unknow method: %s\n', method)
    end

elseif nexps == 3
    if strcmp(method, 'Ours')
        %conf.est.realD.path = '';
        conf.est.realS.path = './results/Our_result_static_rand_motion_3exp'; 
    else
        error('Unknow method: %s\n', method)
    end

else
    error('Unknown nexps: %d\n', nexps)
end



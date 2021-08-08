gpu=${1:-'0'}
python run_model.py --gpu_ids ${gpu} --model hdr2E_flow2s_model \
    --benchmark syn_test_dataset --bm_dir data/HDR_Synthetic_Test_Dataset \
    --mnet_name weight_net --mnet_checkp data/models/CoarseToFine_2Exp/weight_net.pth --fnet_checkp data/models/CoarseToFine_2Exp/flow_net.pth --mnet2_checkp data/models/CoarseToFine_2Exp/refine_net.pth



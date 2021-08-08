gpu=${1:-'0'}
python run_model.py --gpu_ids $gpu --model hdr3E_flow2s_model \
    --benchmark tog13_online_align_dataset --bm_dir data/TOG13_Dynamic_Dataset --test_scene Dog-3Exp-2Stop --align \
    --mnet_name weight_net --fnet_checkp data/models/CoarseToFine_3Exp/flow_net.pth --mnet_checkp data/models/CoarseToFine_3Exp/weight_net.pth --mnet2_checkp data/models/CoarseToFine_3Exp/refine_net.pth 

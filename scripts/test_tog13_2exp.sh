gpu=${1:-'0'}
python run_model.py --gpu_ids ${gpu} --model hdr2E_flow2s_model \
    --benchmark tog13_online_align_dataset --bm_dir data/TOG13_Dynamic_Dataset --align --test_scene ThrowingTowel-2Exp-3Stop \
    --mnet_name weight_net  --fnet_checkp data/models/CoarseToFine_2Exp/flow_net.pth --mnet_checkp data/models/CoarseToFine_2Exp/weight_net.pth --mnet2_checkp data/models/CoarseToFine_2Exp/refine_net.pth 


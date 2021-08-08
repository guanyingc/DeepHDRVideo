gpu=${1:-'0'}
python run_model.py --gpu_ids $gpu --model hdr2E_flow2s_model \
    --benchmark real_benchmark_dataset --bm_dir data/dynamic_RGB_data_2exp_release --test_scene all \
    --mnet_name weight_net  --fnet_checkp data/models/CoarseToFine_2Exp/flow_net.pth --mnet_checkp data/models/CoarseToFine_2Exp/weight_net.pth --mnet2_checkp data/models/CoarseToFine_2Exp/refine_net.pth


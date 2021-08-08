gpu=${1:-'0'}
python main.py --gpu_ids $gpu --model hdr3E_flow2s_model --fnet_checkp data/models/CoarseToFine_3Exp/flow_net.pth --mnet_checkp data/models/CoarseToFine_3Exp/weight_net.pth 


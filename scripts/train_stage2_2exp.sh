gpu=${1:-'0'}
python main.py --gpu_ids $gpu --model hdr2E_flow2s_model --fnet_checkp data/models/CoarseToFine_2Exp/flow_net.pth --mnet_checkp data/models/CoarseToFine_2Exp/weight_net.pth 

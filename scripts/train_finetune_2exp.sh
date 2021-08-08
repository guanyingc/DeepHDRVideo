python main.py --gpu_ids 0,1 --model hdr2E_flow2s_model --up_s1 --init_lr 0.00002 \
    --fnet_checkp data/models/CoarseToFine_2Exp/flow_net.pth --mnet_checkp data/models/CoarseToFine_2Exp/weight_net.pth --mnet2_checkp data/models/CoarseToFine_2Exp/refine_net.pth 

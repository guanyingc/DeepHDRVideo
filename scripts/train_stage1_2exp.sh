gpu=${1:-'0'}
python main.py --gpu_ids $gpu --model hdr2E_flow_model --debug


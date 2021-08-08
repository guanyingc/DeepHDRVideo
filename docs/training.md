# Training Steps

<p align="center">
    <img src='../images/network.jpg' width="900">
</p>

### Code Structure

We use `xxx_model.py` (e.g., `hdr2E_flow_model.py`, `hdr2E_flow2s_model.py`) to control the forward, backward, and visualization procedures. `xxx_model.py` will initialize specific network(s) for training and testing. `xxx_model.py` is placed in `models/`. Network architectures are placed in `models/archs/`.
Due to the limited time for code cleaning, this repository might contain some redundant and confusing codes.

<details>
    <summary>code directory</summary>

```
.
├── datasets
│   ├── hdr_transforms.py
│   ├── __init__.py
│   ├── real_benchmark_dataset.py
│   ├── syn_hdr_dataset.py
│   ├── syn_test_dataset.py
│   ├── syn_vimeo_dataset.py
│   ├── tog13_online_align_dataset.py
│   └── tog13_prealign_dataset.py
├── extensions
│   ├── dcn/
│   └── pytorch_msssim/
├── main.py
├── matlab
│   ├── config_eval.m
│   ├── get_expo_types.m
│   ├── get_hdr_list.m
│   ├── get_summary_filename.m
│   ├── Library/
│   ├── main_eval.m
│   ├── merge_res_same_index.m
│   ├── mulog_tonemap.m
│   ├── my_psnr.m
│   ├── paral_eval_HDRs.m
│   └── save_hdr_txt_results.m
├── models
│   ├── archs/
│   │   ├── DAHDRnet.py
│   │   ├── edvr_networks.py
│   │   ├── flow_networks.py
│   │   ├── __init__.py
│   │   ├── weight_EG19net.py
│   │   └── weight_net.py
│   ├── base_model.py
│   ├── hdr2E_flow2s_model.py
│   ├── hdr2E_flow_model.py
│   ├── hdr2E_model.py
│   ├── hdr3E_flow2s_model.py
│   ├── hdr3E_flow_model.py
│   ├── __init__.py
│   ├── losses.py
│   ├── model_utils.py
│   ├── network_utils.py
│   └── noise_utils.py
├── options
│   ├── base_opts.py
│   ├── __init__.py
│   ├── run_model_opts.py
│   └── train_opts.py
├── README.md
├── requirements.txt
├── run_model.py
└── utils
    ├── clean.sh
    ├── compute_nbr_trans_for_video.py
    ├── eval_utils.py
    ├── image_utils.py
    ├── __init__.py
    ├── logger.py
    ├── recorders.py
    ├── test_utils.py
    ├── time_utils.py
    ├── tonemapper.py
    ├── train_utils.py
    └── utils.py

```
</details>

### Prepare the training data
Please first go through [DeepHDRVideo-Dataset](https://github.com/guanyingc/DeepHDRVideo-Dataset) to familiarize yourself with the training datast.
Download `Synthetic_Train_Data_HdM-HDR-2014.tgz`, `Synthetic_Train_Data_LiU_HDRv.tgz`, `vimeo_septuplet.tgz` from Google Drive (`Synthetic_Dataset/`) and unzip them in `data/`.

### Step 1: Train CoarseNet
```shell
# Train two-exposure model
python main.py --gpu_ids 0 --model hdr2E_flow_model

# Train three-exposure model
python main.py --gpu_ids 0 --model hdr3E_flow_model

# Results and checkpoints can be found in logdir/syn_vimeo_dataset/ICCV/
```

### Step 2: Train RefineNet
```shell
# Train two-exposure model
python main.py --gpu_ids 0 --model hdr2E_flow2s_model --fnet_checkp /path/to/flow_net.pth --mnet_checkp /path/to/weight_net.pth 

# Train three-exposure model
python main.py --gpu_ids 0 --model hdr3E_flow2s_model --fnet_checkp /path/to/flow_net.pth --mnet_checkp /path/to/weight_net.pth 

# Results and checkpoints can be found in logdir/syn_vimeo_dataset/ICCV/
```

### Step 3: End-to-end finetuning
```shell
# Train two-exposure model, update stage1, using a smaller learning rate
python main.py --gpu_ids 0,1 --model hdr2E_flow2s_model --up_s1 --init_lr 0.00002 \
    --fnet_checkp /path/to/flow_net.pth --mnet_checkp /path/to/weight_net.pth --mnet2_checkp /path/to/refine_net.pth 
```

#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# gradio demo executable
# --------------------------------------------------------
import os
import torch
from glob import glob

from mast3r.glomap_local import get_args_parser, reconstruct_scene
from mast3r.model import AsymmetricMASt3R

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    weights_path = args.weights

    model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)

    # list files
    filelist = glob(os.path.join(args.input_dir, '*'))
    # sort the images according to the number present in them. e.g.: frame_1.jpg, frame_3.jpg, frame_27.jpg, (...)
    filelist = sorted(filelist, key=lambda f: int(os.path.splitext(os.path.basename(f))[0].replace("frame_", "")))
    # grab only the first N files, and downsample them at a specific rate
    filelist = filelist[0 : args.first_n : args.downsample_rate]

    # options: "complete", "swin", "logwin", "oneref"
    scenegraph_type = "complete"
    winsize = 1
    win_cyclic = True
    refid = 0
    shared_intrinsics = True

    cache_path = args.tmp_dir
    os.makedirs(cache_path, exist_ok=True)
    reconstruct_scene(args.glomap_bin, cache_path, model, args.retrieval_model, args.device, args.image_size,
                      filelist, scenegraph_type, winsize, win_cyclic, refid, shared_intrinsics)
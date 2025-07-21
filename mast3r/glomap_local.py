#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# gradio demo functions
# --------------------------------------------------------
import pycolmap
import os
import copy
import shutil
import torch

from kapture.converter.colmap.database_extra import kapture_to_colmap
from kapture.converter.colmap.database import COLMAPDatabase

from mast3r.colmap.mapping import kapture_import_image_folder_or_list, run_mast3r_matching, glomap_run_mapper
from mast3r.retrieval.processor import Retriever
from mast3r.image_pairs import make_pairs

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import load_images
from dust3r.demo import get_args_parser as dust3r_get_args_parser

# Type hints
from mast3r.model import AsymmetricMASt3R
from typing import Union, List
from kapture import CameraType

def get_args_parser():
    # Inserting a method to print the available options in the Enum without changing directly in the library
    def camera_type_options(cls):
        return [member.name for member in cls]
    CameraType.options = classmethod(camera_type_options)

    parser = dust3r_get_args_parser()
    parser.add_argument('--glomap_bin', default='glomap', type=str, help='glomap bin')
    parser.add_argument('--input_dir', type=str, help='path to the input directory containing the images to be processed')
    parser.add_argument('--first_n', type=int, default=-1, help='load the first N images contained in the input dir')
    parser.add_argument('--downsample_rate', type=int, default=1, help='drop files at the given rate')
    parser.add_argument('--retrieval_model', default=None, type=str, help="retrieval_model to be loaded")
    parser.add_argument('--camera_model', default=CameraType.UNKNOWN_CAMERA.name, type=str, choices=CameraType.options(), help="Model of the camera used to capture the images")
    parser.add_argument('--camera_params', nargs="*", default=[], type=float, help="Parameters of the camera model. If the camera model is unknown then this argument can be ommited")

    actions = parser._actions
    for action in actions:
        if action.dest == 'model_name':
            action.choices = ["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"]
    # change defaults
    parser.prog = 'mast3r demo'
    return parser

def reconstruct_scene(glomap_bin : str, 
                      outdir : str, 
                      model : AsymmetricMASt3R, 
                      retrieval_model : str, 
                      device : str, 
                      image_size : int,
                      filelist: Union[str, list], 
                      scenegraph_type: str, 
                      winsize: int,
                      win_cyclic: bool, 
                      refid: int, 
                      shared_intrinsics: bool, 
                      camera_model : CameraType = CameraType.UNKNOWN_CAMERA,
                      camera_params : List[float] = [],
                      glomap_only : bool = False):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=False)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0]]

    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    elif scenegraph_type == "retrieval":
        scene_graph_params.append(str(winsize))  # Na
        scene_graph_params.append(str(refid))  # k

    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append('noncyclic')
    scene_graph = '-'.join(scene_graph_params)

    sim_matrix = None
    if 'retrieval' in scenegraph_type:
        assert retrieval_model is not None
        retriever = Retriever(retrieval_model, backbone=model, device=device)
        with torch.no_grad():
            sim_matrix = retriever(filelist)

        # Cleanup
        del retriever
        torch.cuda.empty_cache()

    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True, sim_mat=sim_matrix)

    cache_dir = os.path.join(outdir, 'cache')

    root_path = os.path.commonpath(filelist)
    filelist_relpath = [
        os.path.relpath(filename, root_path).replace('\\', '/')
        for filename in filelist
    ]
    kdata = kapture_import_image_folder_or_list((root_path, filelist_relpath), shared_intrinsics, camera_model, camera_params)
    image_pairs = [
        (filelist_relpath[img1['idx']], filelist_relpath[img2['idx']])
        for img1, img2 in pairs
    ]

    colmap_db_path = os.path.join(cache_dir, 'colmap.db')
    if not glomap_only:
        if os.path.isfile(colmap_db_path):
            os.remove(colmap_db_path)

        os.makedirs(os.path.dirname(colmap_db_path), exist_ok=True)
        colmap_db = COLMAPDatabase.connect(colmap_db_path)
        try:
            kapture_to_colmap(kdata, root_path, tar_handler=None, database=colmap_db,
                            keypoints_type=None, descriptors_type=None, export_two_view_geometry=False)
            colmap_image_pairs = run_mast3r_matching(model, image_size, 16, device,
                                                    kdata, root_path, image_pairs, colmap_db,
                                                    False, 5, 1.001,
                                                    False, 3)
            colmap_db.close()
        except Exception as e:
            print(f'Error {e}')
            colmap_db.close()
            exit(1)

        if len(colmap_image_pairs) == 0:
            raise Exception("no matches were kept")

        # colmap db is now full, run colmap
        print("verify_matches")
        f = open(cache_dir + '/pairs.txt', "w")
        for image_path1, image_path2 in colmap_image_pairs:
            f.write("{} {}\n".format(image_path1, image_path2))
        f.close()
        pycolmap.verify_matches(colmap_db_path, cache_dir + '/pairs.txt')

    reconstruction_path = os.path.join(cache_dir, "reconstruction")
    if os.path.isdir(reconstruction_path):
        shutil.rmtree(reconstruction_path)
    os.makedirs(reconstruction_path, exist_ok=True)
    glomap_run_mapper(glomap_bin, colmap_db_path, reconstruction_path, root_path, optimize_intrinsics=(camera_model == CameraType.UNKNOWN_CAMERA))

    outfile_name = os.path.join(outdir, "scene.ply")

    ouput_recon = pycolmap.Reconstruction(os.path.join(reconstruction_path, '0'))
    ouput_recon.export_PLY(outfile_name)   
    # cam_T_world = ouput_recon.image(1).cam_from_world
    # cam_S_world = pycolmap.Sim3d(1., cam_T_world.rotation, cam_T_world.translation)
    # ouput_recon.transform(cam_S_world)
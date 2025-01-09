# -*- coding: utf-8 -*-
"""
This Script Preprocesses Sentinel-2 data for AGB Estimation.
"""
"""
@Time    : 05/09/2024 18:06
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : run_preprocessing
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
sys.path.append(r'Your_src_Directory')
from src.preprocessing import preprocessing as prep

if __name__ == '__main__':

    ## Set variables
    sentinel2_path = r'E:\Data\s4g_assignment\Sentinel2_Data\GRANULE\L2A_T36NYF_A047751_20240813T080642\IMG_DATA\R20m'
    output_stack_path = r"E:\Data\s4g_assignment\Sentinel2_stacks"
    bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
    labels_path = "E:\Data\s4g_assignment\Kenya_AGB.geojson"
    aoi_mask_path = "E:\Data\s4g_assignment\Kenya_AOI.geojson"

    ########
    ## init
    ########
    data_preprocessing = prep(sentinel2_path,output_stack_path,bands, labels_path)

    ########
    ## stack & crop Sentienl-2 data
    ########
    stack_file_path = data_preprocessing.stack_sentinel()
    output_cropped_path = data_preprocessing.crop_stack_to_aoi(stack_file_path, aoi_mask_path, output_stack_path)

    ########
    ## convert labels from vector to raster
    ########
    output_label_path = data_preprocessing.convert_labels_to_raster(output_cropped_path, output_stack_path)

    ########
    ## Compute Indices
    ########
    output_trad_ind_stack = data_preprocessing.compute_traditional_indices(output_cropped_path)
    output_stack_vi_path = data_preprocessing.compute_red_edge_indices(output_trad_ind_stack)

    ########
    ## Normlaise dataset
    ########
    normalised_stack_path = data_preprocessing.normalise_stack(output_stack_vi_path)

    ########
    ## add labels to dataset
    ########
    stacked_with_labels_path = data_preprocessing.stack_label(normalised_stack_path, output_label_path)

# -*- coding: utf-8 -*-
"""
This script trains and anlyses an XGBoost model for AGB estimation.
"""
"""
@Time    : 06/09/2024 12:11
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : run_model
"""

from src.xgboost import XGBoostModel as xgb

stack_path = r"E:\Data\s4g_assignment\Sentinel2_stacks\sen2_stack_re_labels.tif" #stack_with_labels.tif"#
results_path = r"E:\Data\s4g_assignment\Results"



if __name__ == '__main__':
    #########
    ## model init
    #########
    model = xgb(stack_path)

    #########
    ## data prep
    #########
    model.load_raster_data()
    model.prepare_data()

    #########
    ## hyperparam tuning
    #########
    best_params, best_score = model.tune_hyperparameters()

    #########
    ## training
    #########
    # model.train_model()
    model.train_model_with_eval()

    #########
    ## model eval
    #########
    model.evaluate_model()

    #########
    ## feature importance
    #########
    model.plot_feature_importance()

    #########
    ## plot prediction v observed
    #########
    model.plot_predicted_vs_observed()

    #########
    ## inference
    #########
    model.generate_and_save_map(results_path,)










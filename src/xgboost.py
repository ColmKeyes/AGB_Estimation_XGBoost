# -*- coding: utf-8 -*-
"""
This script provides functions to train, test and analyse an XGBoost model for AGB Estimation.
"""
"""
@Time    : 06/09/2024 12:05
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : xgboost
"""
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import os


class XGBoostModel:
    def __init__(self, raster_path, label_band_idx=-1, test_size=0.2, random_state=42, n_estimators=None, learning_rate=0.1, max_depth=5):
        self.raster_path = raster_path
        self.label_band_idx = label_band_idx
        self.test_size = test_size
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=self.n_estimators,
                                      learning_rate=self.learning_rate, max_depth=self.max_depth)
        self.X = None
        self.y = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None



    def load_raster_data(self):

        # Load & flatten
        with rasterio.open(self.raster_path) as src:
            data = src.read().astype('float32')
            spectral_bands = data[:self.label_band_idx]
            labels = data[self.label_band_idx]
            self.X = spectral_bands.reshape(spectral_bands.shape[0], -1).T
            self.y = labels.flatten()

    def prepare_data(self):

        # Mask out nan pixels for training
        valid_mask = ~np.isnan(self.y)
        X = self.X[valid_mask]
        y = self.y[valid_mask]

        # convert any zero to nan for no-data handling
        X = np.where(X == 0, np.nan, X)

        # train-test-val split, 70/15/15
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=self.random_state)
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=self.random_state)
        self.X_train, self.X_valid, self.X_test = X_train, X_valid, X_test
        self.y_train, self.y_valid, self.y_test = y_train, y_valid, y_test

    ###########
    ##  Hyperparameter tuning
    ###########
    def tune_hyperparameters(self):
        """
        Perform k-fold cross-validation to tune hyperparameters.
        """
        param_grid = {
            'n_estimators': [100,200,500],
            'max_depth': [3,5],
            'learning_rate': [.01],
            'subsample': [0.8],
            'colsample_bytree':[ 0.8],
            'alpha': [1,5,10],
            'lambda': [1,5,10]
        }

        # Initialise XGBoost Regressor
        xgb_model = xgb.XGBRegressor()

        # Set up grid search
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            verbose=1)

        # Perform the grid search on training data
        grid_search.fit(self.X_train, self.y_train)

        # Log parameters and results
        print(f"Best hyperparameters found: {grid_search.best_params_}")
        print(f"Best cross-validation score (negative MSE): {grid_search.best_score_}")

        # Store results
        results = grid_search.cv_results_
        print("\nCross-validation results:")
        for i, params in enumerate(results['params']):
            print(f"Params: {params}, Score: {results['mean_test_score'][i]}")

        # Set model params to best
        self.model = grid_search.best_estimator_

        print(f"Best hyperparameters found: {grid_search.best_params_}")
        print(f"Best cross-validation score (negative MSE): {grid_search.best_score_}")
        return grid_search.best_params_, grid_search.best_score_

    ###########
    ## simple training
    ###########
    def train_model(self):
        if self.X_train is None or self.y_train is None:
            self.prepare_data()
        self.model.fit(self.X_train, self.y_train, eval_set=[(self.X_valid, self.y_valid)], early_stopping_rounds=10, verbose=True)

    ###########
    ## advanced training & eval on validation data
    ###########
    def train_model_with_eval(self, early_stopping_rounds=10, eval_metric='rmse'):
        """
        Train the XGBoost model using early stopping and track evaluation metrics.
        """
        if self.X_train is None or self.y_train is None:
            self.prepare_data()

        # Convert data to dmatrix, required for training xgboost
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dvalid = xgb.DMatrix(self.X_valid, label=self.y_valid)

        # get params
        params = self.model.get_params()
        params.update({
            'objective': 'reg:squarederror',
            'eval_metric': eval_metric,})
        evals = [(dtrain, 'train'), (dvalid, 'eval')]
        evals_result = {}

        # Train with early stopping based on validation set performance
        self.model = xgb.train(params, dtrain, num_boost_round=1000,#params.get('n_estimators'),
                               evals=evals, early_stopping_rounds=early_stopping_rounds,
                               verbose_eval=True, evals_result=evals_result)
        print("Training metrics: ", evals_result['train'][eval_metric])
        print("Validation metrics: ", evals_result['eval'][eval_metric])

        # Plotting metrics
        epochs = len(evals_result['train'][eval_metric])
        x_axis = range(0, epochs)
        plt.plot(x_axis, evals_result['train'][eval_metric], label='Train')
        plt.plot(x_axis, evals_result['eval'][eval_metric], label='Validation')
        plt.xlabel('Boosting Rounds')
        plt.ylabel(eval_metric.upper())
        plt.title(f'XGBoost {eval_metric.upper()} over Rounds')
        plt.grid(True)
        plt.legend()
        plt.show()

    ###########
    ## Evaluate model on test data
    ###########
    def evaluate_model(self):
        if self.X_test is None or self.y_test is None:
            self.prepare_data()

        dtest = xgb.DMatrix(self.X_test)
        y_pred = self.model.predict(dtest)

        r2 = r2_score(self.y_test, y_pred)
        print(f"R² score: {r2}")

        mse = mean_squared_error(self.y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        return mse, r2


    ###########
    ## feature importance plot
    ###########
    def plot_feature_importance(self):
        feature_names = ['Blue', 'Green', 'Red', 'Red Edge 1', 'Red Edge 2',
                          'Red Edge 3', 'NIR', 'SWIR 1', 'SWIR 2',
                         'CIre', 'IRECI', 'MCARI', 'NDVIre1', 'NDVIre2', 'NDVIre3', 'NDre1', 'NDre2', 'SRre',
                         'ARVI', 'CIg', 'DVI', 'EVI', 'GNDVI', 'MSAVI', 'NDII', 'NDVI', 'SR']

        # Get feature importance
        importance_dict = self.model.get_score(importance_type='weight')

        # Map importance to feature names
        importance_named = {}
        for k, v in importance_dict.items():
            feature_index = int(k[1:])
            if feature_index < len(feature_names):
                importance_named[feature_names[feature_index]] = v
        sorted_importance = dict(sorted(importance_named.items(), key=lambda item: item[1], reverse=True))

        # Plot
        plt.barh(list(sorted_importance.keys()), list(sorted_importance.values()))
        plt.title("Feature Importance")
        plt.xlabel("Importance Score")
        # plt.grid(True)
        plt.show()
        plt.pause(10)

    ###########
    ## plot observed vs predicted biomass
    ###########
    def plot_predicted_vs_observed(self):

        # Get predictions on training data
        dmatrix_xtrain = xgb.DMatrix(self.X_train)
        predictions = self.model.predict(dmatrix_xtrain)

        # Scatter plot of predicted vs observed AGB (y_train)
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_train, predictions, edgecolors='black', facecolors='none', marker='o', label='Predicted vs Observed')

        # Fit a line to the data
        m, b = np.polyfit(self.y_train, predictions, 1)  # Fit a linear regression
        plt.plot(self.y_train, m * np.array(self.y_train) + b, color='red', linestyle= "--")

        min_val = min(min(self.y_train), min(predictions))
        max_val = max(max(self.y_train), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='-')

        # Labels and title
        plt.xlabel('Observed AGB (ton/ha)')
        plt.ylabel('Predicted AGB (ton/ha)')
        plt.title('Predicted vs Observed AGB')
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()

    ###########
    ## Inference
    ###########
    def generate_and_save_map(self, output_path):

        pred_output_path = os.path.join(output_path, 'pred_agb_map.tif')

        # Convert X to DMatrix for prediction
        dmatrix_new = xgb.DMatrix(self.X)

        # Generate predictions
        predictions = self.model.predict(dmatrix_new)

        # Open the original raster to get the dimensions
        with rasterio.open(self.raster_path) as src:
            meta = src.meta
            image_height = src.height
            image_width = src.width

        # Reshape the predictions to the original image dimensions
        predicted_map = predictions.reshape(image_height, image_width)

        # Save the predicted map as a GeoTIFF file
        meta.update(dtype='float32', count=1)

        with rasterio.open(pred_output_path, 'w', **meta) as dst:
            dst.write(predicted_map.astype('float32'), 1)

    ###########
    ## Save & load Model
    ###########
    def save_model(self, model_path='xgboost_model.json'):
        self.model.save_model(model_path)

    def load_model(self, model_path='xgboost_model.json'):
        self.model.load_model(model_path)



# AGB Estimation using XGBoost

A machine learning project for estimating Above Ground Biomass (AGB) using Sentinel-2 satellite imagery and XGBoost regression. This project processes multi-spectral satellite data, computes vegetation indices, and trains a predictive model to estimate forest biomass in Kenya.

## 🌟 Overview

This project implements an end-to-end pipeline for AGB estimation using:
- **Sentinel-2 satellite imagery** for multi-spectral data
- **Traditional and Red-Edge vegetation indices** for enhanced feature extraction
- **XGBoost regression** for biomass prediction
- **Comprehensive preprocessing** including normalization and data preparation
- **Model evaluation** with feature importance analysis and visualization

## 📁 Project Structure

```
AGB_Estimation_XGBoost/
├── bin/
│   ├── run_preprocessing.py    # Data preprocessing pipeline
│   ├── run_model.py           # Model training and evaluation
│   └── output_map.html        # Interactive map visualization
├── src/
│   ├── preprocessing.py       # Preprocessing utilities
│   └── xgboost.py            # XGBoost model implementation
├── environment_yaml.yaml     # Conda environment specification
└── README.md                 # This file
```

## 🚀 Features

### Data Preprocessing
- **Multi-band stacking** of Sentinel-2 imagery (9 spectral bands)
- **AOI cropping** using vector boundaries
- **Label rasterization** from vector AGB data
- **Vegetation indices computation**:
  - Traditional indices: ARVI, CIg, DVI, EVI, GNDVI, MSAVI, NDII, NDVI, SR
  - Red-edge indices: CIre, IRECI, MCARI, NDVIre1-3, NDre1-2, SRre
- **Data normalization** to [0,1] range
- **Label integration** for supervised learning

### Machine Learning Pipeline
- **Hyperparameter tuning** using GridSearchCV with 5-fold cross-validation
- **Train/validation/test split** (70/15/15)
- **Early stopping** to prevent overfitting
- **Model evaluation** with R² and MSE metrics
- **Feature importance analysis**
- **Prediction visualization** (observed vs predicted)
- **Spatial prediction mapping**

## 🛠️ Installation

### Prerequisites
- Anaconda or Miniconda
- Python 3.9+

### Environment Setup
1. Clone the repository:
```bash
git clone https://github.com/ColmKeyes/AGB_Estimation_XGBoost.git
cd AGB_Estimation_XGBoost
```

2. Create and activate the conda environment:
```bash
conda env create -f environment_yaml.yaml
conda activate S4G_assignment
```

### Key Dependencies
- **XGBoost**: Machine learning framework
- **Scikit-learn**: Model evaluation and preprocessing
- **Rasterio**: Geospatial raster processing
- **GeoPandas**: Vector data handling
- **Matplotlib**: Visualization
- **NumPy/Pandas**: Data manipulation
- **Jupyter**: Interactive development
- **Geemap**: Earth Engine integration

## 📊 Usage

### 1. Data Preprocessing
Run the preprocessing pipeline to prepare Sentinel-2 data:

```bash
python bin/run_preprocessing.py
```

**Required inputs:**
- Sentinel-2 L2A imagery (20m resolution)
- AGB reference data (GeoJSON format)
- Area of Interest boundary (GeoJSON format)

**Preprocessing steps:**
1. Stack Sentinel-2 bands (B02, B03, B04, B05, B06, B07, B8A, B11, B12)
2. Crop to study area
3. Convert vector labels to raster
4. Compute vegetation indices
5. Normalize data
6. Integrate labels with features

### 2. Model Training and Evaluation
Train the XGBoost model and evaluate performance:

```bash
python bin/run_model.py
```

**Training pipeline:**
1. Load preprocessed raster stack
2. Prepare training/validation/test datasets
3. Hyperparameter optimization
4. Model training with early stopping
5. Performance evaluation
6. Feature importance analysis
7. Generate prediction maps

### 3. Model Configuration
Key hyperparameters optimized:
- `n_estimators`: [100, 200, 500]
- `max_depth`: [3, 5]
- `learning_rate`: [0.01]
- `subsample`: [0.8]
- `colsample_bytree`: [0.8]
- `alpha`: [1, 5, 10] (L1 regularization)
- `lambda`: [1, 5, 10] (L2 regularization)

## 📈 Model Performance

The model provides:
- **R² score** for goodness of fit assessment
- **Mean Squared Error (MSE)** for prediction accuracy
- **Feature importance rankings** showing most predictive variables
- **Training/validation curves** for overfitting detection
- **Predicted vs observed plots** for model validation

## 🗺️ Outputs

### Generated Files
- `pred_agb_map.tif`: Spatial AGB prediction map
- `output_map.html`: Interactive web map visualization
- Training metrics and evaluation plots
- Feature importance rankings

### Visualization Features
- Interactive map with layer controls
- Prediction uncertainty visualization
- Feature importance bar charts
- Model performance plots

## 🌍 Study Area

The project focuses on forest biomass estimation in **Kenya**, utilizing:
- **Sentinel-2 L2A** surface reflectance data
- **20m spatial resolution** for detailed analysis
- **Multi-temporal** capability for seasonal analysis
- **Red-edge bands** for enhanced vegetation monitoring

## 📝 Methodology

### Spectral Bands Used
| Band | Wavelength | Resolution | Description |
|------|------------|------------|-------------|
| B02  | 490nm      | 20m        | Blue |
| B03  | 560nm      | 20m        | Green |
| B04  | 665nm      | 20m        | Red |
| B05  | 705nm      | 20m        | Red Edge 1 |
| B06  | 740nm      | 20m        | Red Edge 2 |
| B07  | 783nm      | 20m        | Red Edge 3 |
| B8A  | 865nm      | 20m        | NIR |
| B11  | 1610nm     | 20m        | SWIR 1 |
| B12  | 2190nm     | 20m        | SWIR 2 |

### Vegetation Indices
**Traditional Indices:**
- ARVI, CIg, DVI, EVI, GNDVI, MSAVI, NDII, NDVI, SR

**Red-Edge Indices:**
- CIre, IRECI, MCARI, NDVIre1, NDVIre2, NDVIre3, NDre1, NDre2, SRre

## 🔧 Customization

### Modifying Input Paths
Update paths in `bin/run_preprocessing.py` and `bin/run_model.py`:
```python
sentinel2_path = "path/to/sentinel2/data"
labels_path = "path/to/agb/labels.geojson"
aoi_mask_path = "path/to/aoi/boundary.geojson"
results_path = "path/to/output/directory"
```

### Adjusting Model Parameters
Modify hyperparameter grid in `src/xgboost.py`:
```python
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    # Add more parameters as needed
}
```

## 📚 References

- Sentinel-2 User Handbook: ESA
- XGBoost Documentation: https://xgboost.readthedocs.io/
- Remote Sensing of Forest Biomass: A Review

## 👨‍💻 Author

**Colm Keyes**
- Email: keyesco@tcd.ie
- GitHub: [@ColmKeyes](https://github.com/ColmKeyes)

## 📄 License

This project is available under the MIT License. See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For questions or issues, please:
1. Check the documentation
2. Open an issue on GitHub
3. Contact the author via email

---

*This project was developed for forest biomass estimation research using satellite remote sensing and machine learning techniques.*

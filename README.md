# soil-thermal-diffusivity-ml
MATLAB codes for MLP, Random Forest, XGBoost, and MLR for estimating soil thermal diffusivity
# MATLAB Codes for Soil Thermal Diffusivity Prediction

This repository contains MATLAB codes for estimating soil thermal diffusivity using three machine learning models and one statistical regression method.

## Models / Methods Included

- **MLP** (Multilayer Perceptron / Neural Network) - Machine Learning
- **Random Forest** (RF) - Machine Learning
- **XGBoost** - Machine Learning
- **Multiple Linear Regression (MLR)** - Statistical Method (not machine learning)

## Paper Information

These codes accompany the paper:

> "A Hybrid Machine Learning–Physics Approach for Retrieving Thermal Diffusivity, Simulating Soil Temperature, and Zoning Thermal Regimes in Iran"

## Requirements

- MATLAB R2020b or later
- Statistics and Machine Learning Toolbox
- Deep Learning Toolbox (for MLP)
- Python with XGBoost installed (for XGBoost model)

## Data Format

The training data should be an Excel file with the following columns:

| Column | Description |
|--------|-------------|
| ndvi | NDVI values |
| sstad | SSTA day |
| sstay | SSTA year |
| bd | Bulk density |
| clay | Clay percentage |
| sand | Sand percentage |
| silt | Silt percentage |
| soc | Soil organic carbon |
| wv | Water vapor |
| alfa | Thermal diffusivity (target variable) |

## How to Run

1. Clone this repository
2. Place your training data Excel file in the `data/` folder
3. Place your GeoTIFF rasters for regional prediction in the input path
4. Open MATLAB and navigate to the repository folder
5. Run each script individually

## Outputs

Each model/method generates:
- Performance metrics (R², RMSE, MAE, NSE, PBIAS)
- Publication-quality figures
- Saved model file (.mat)
- Regional prediction map (GeoTIFF)

## Note

- MLP, RF, and XGBoost are machine learning models
- MLR (Multiple Linear Regression) is a classical statistical method included for comparison

## GitHub

https://github.com/86311972

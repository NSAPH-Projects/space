# SpaCE examples

This folder contains example scripts:

* **[ols](./ols.py)**: causal average effect estimation of the binary treatment using ordinary least squares (OLS) treatment effects estimation. 

## Causal Inference Evaluation

```
conda env create --file ../requirements.yaml 
conda activate space-env
curl -sSL https://raw.githubusercontent.com/NSAPH-Projects/space/master/data/data.sh | bash -s XGBOOST BINARY 12345
pip install causalinference
python linreg --dataset "./medisynth-xgboost-binary-sample.csv"
```
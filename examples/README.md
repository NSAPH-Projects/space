# SpaCE examples

This folder contains example scripts:

* **[linreg](./linreg.py)**: causal average effect estimation of the binary treatment using a simple linear regression. 

## Causal Inference Evaluation

```
conda env create --file ../requirements.yaml 
conda activate space-env
curl -sSL https://raw.githubusercontent.com/NSAPH-Projects/space/master/data/data.sh | bash -s XGBOOST BINARY 12345
python linreg --space_dataset "./space_dataset.csv"
```
# SpaCE: The Spatial Confounding (Benchmarking) Environment

[![](<https://img.shields.io/badge/Dataverse-10.7910/DVN/SYNPBS-orange>)](https://www.doi.org/10.7910/DVN/SYNPBS)

## 🚀 Description

**SpaCE**: The Spatial Confounding Environment is a benchmarking dataset for causal inference that incorporartes spatial confounding. The **SpaCE** datasets contain real confounder and exposure/treatment data inspired by environmental health studies. The synthetic outcome and counterfactual are generated according to recommended practices for causal evaluation by mimicking the real outcome data distribution learned with machine learning and neural network methods. Spatial confounding is achieved by masking influential confounders in the learned model. 

## 🛰️ Code and methods

The code for source data processing and outcome modeling can be found in the `analysis` directory. The code for the **SpaCE** benchmarking data retrieval and sampling is in the `data` directory. The code benchmarking dataset with all confounders can be viewed on the Harvard Dataverse repository for transparency and reproducibility.

## 🧑‍🚀 The API

To retrieve and generate the data, run the command below in the Terminal: 

``` sh
curl -sSL https://raw.githubusercontent.com/NSAPH-Projects/space/master/data/data.sh | bash -s NN|XGBOOST SEED PATH
```

The input parameters of the command above are:

| Parameter           | Meaning            |
|--------------------|-------------------|
| method          |   NN or XGBOOST |
| random seed         | integer              |
| path (optional)  |    output file path  |

The command downloads the core data and uses it together with the user's input to generate a sample of the potential outcomes (factual and counterfactual) from the core data and the model predictions. The sampling mechanism is tuned to mimic the variability in the observed data.


## 🔭 Data dictionary

The data dictionary for the **SpaCE** benchmark data:

| Variable           | Values            |
|--------------------|-------------------|
| treatment          |                   |
| covariates         |                   |
| synthetic_outcome  |                   |
| models             | xgboost, nn_vcnet |
| error_gp_kernel    |                   |
| predictions        |                   |
| counterfactuals    |                   |
| spatial_smoothness |                   |
| importance         |                   |

## 👽 Contact

We welcome contributions and feedback about **SpaCE**. If you have any suggestions or ideas, please open an issue or submit a pull request. Thank you for your interest in our data.

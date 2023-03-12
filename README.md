# SpaCE
SpaCE, the Spatial Confounding Environment, loads benchmark datasets for causal inference methods tackling spatial confounding.

[![](<https://img.shields.io/badge/Dataverse-10.7910/DVN/SYNPBS-orange>)](https://www.doi.org/10.7910/DVN/SYNPBS)

## 🚀 Description

**SpaCE**: The Spatial Confounding Environment loads benchmark datasets for causal inference that incorporartes spatial confounding. The **SpaCE** datasets contain real confounder and exposure/treatment data inspired by environmental health studies. The synthetic outcome and counterfactuals are generated for causal evaluation. They mimick real outcome data distributions learned with machine learning and neural network methods. Spatial confounding is achieved by masking influential confounders in the learned model. 

## 🛰️ Code and methods

The code for **SpaCE** data loaders is in the `space` directory. The **SpaCE** datasets are found in the [Harvard Dataverse repository](https://dataverse.harvard.edu/) for transparency and reproducibility.

Code examples with outcome modeling can be found in the `examples` directory. 

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


## 🔭 Attributes

 **SpaCE** benchmark data:

| Atributes          |
|--------------------|
| treatment          |
| covariates         |
| synthetic_outcome  |
| counterfactuals    |

| Metadta            | Values            |
|--------------------|-------------------|
| predictive_model   | xgboost, nn_vcnet |
| error_model        |                   |


## 👽 Contact

We welcome contributions and feedback about **SpaCE**. If you have any suggestions or ideas, please open an issue or submit a pull request. Thank you for your interest in our data.

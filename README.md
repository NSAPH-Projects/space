
<!-- 
> _This project is in active development. The API is not yet stable.</text> -->


![](resources/logo.png)

[![](<https://img.shields.io/badge/Dataverse-10.7910/DVN/SYNPBS-orange>)](https://www.doi.org/10.7910/DVN/SYNPBS)
[![Licence](https://img.shields.io/pypi/l/spacebench.svg)](https://pypi.org/project/spacebench)
[![PyPI version](https://img.shields.io/pypi/v/spacebench.svg)](https://pypi.org/project/spacebench)
[![codecov](https://codecov.io/gh/NSAPH-Projects/space/branch/dev/graph/badge.svg?token=I4BDXHGRFR)](https://codecov.io/gh/NSAPH-Projects/space)


## 🚀 Description

SpaCE loads benchmark datasets for causal inference that incorporartes spatial confounding. The **SpaCE** datasets contain real confounder and exposure/treatment data inspired by environmental health studies. The synthetic outcome and counterfactuals are generated for causal evaluation. They mimick real outcome data distributions learned with machine learning and neural network methods. Spatial confounding is achieved by masking influential confounders in the learned model. 


## 🐍 Installation

Install PyPI version:

```sh
pip install spacebench
```

Or the latest version from GitHub:

``` sh
pip install git+https://github.com/NSAPH-Projects/space
```

For access to the 🔥 features install the development version:

``` sh
pip install git+https://github.com/NSAPH-Projects/space@dev
```

We recommend Python 3.10 or higher. See the [docs](https://nsaph-projects.github.io/space/) and `requirements.txt` for more information.

## 🐢 Getting started


## 🙉 Code of Conduct

Please note that the SpaCE project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By contributing to this project, you agree to abide by its terms.

## 👽 Contact

We welcome contributions and feedback about **spacebench**. If you have any suggestions or ideas, please open an issue or submit a pull request. Thank you for your interest in our data.

## Documentation


## Citation

``` bibtex
@misc{space,
  author = {Tec, Mauricio AND Ana Trisovic AND Michelle Audirac AND Sophie Woodward AND Kate Hu AND Naeem Khoshnevis AND Francesca Dominici},
  title = {SpaCE: The Spatial Confounding Environment},
  year = {2023},
  note = {Github repository: \url{https://github.com/NSAPH-Projects/space}},
}
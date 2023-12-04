## spacebench (developing)

### Added

- Use new covariate groups and confounding scores per metric. Confounding scores are now a dictionary with keys ate, erf, ite, and importance. The first three correspond to the error scores (absolute ate error, mean absolute erf error, mean root-mean-squared ite error) using a baseline model from autogluon. The importance scores correspond to the max of min(outcome_importance, treatment_importance), which are the previously used scores.
- Updated `env.py` to allow for different data formats since data collections in `space-data` can now use `.parquets`, `.graphmlz` and other compressed formats to make read speed and storage more efficient.

### Changed

- Fixed bug in Space Dataset where `smoothness_score` was called `snoothness scores``.
- `confounding_score` and `smoothness_score` are not both singular in a SpaceDataset. 
- Mask entire covariate groups from new covariate groupping in the space environments.
  
### Removed

- Remove option to filter by confounding score in make. It is not useful since user can filter externally examining the `confounding_scores` attribute.
  

## v0.0.2

### Added

- DAPSm algorithm. Basic functionality with optimal and greedy matching.
- Evaluator classes with demo on evaluator_demo.ipynb
- SpaceEnv classes (same demo). Data automatically downloads given name.
- Added Masterfile but needs to be updated with new datasets.
- `list_envs` in DataMaster (previously `list_datasets`) now has an option for `binary=True` or `continuous=True`
- Added examples of benchmarks in the `examples/` folder.
- Add nice printing methods for SpaceDataset and SpaceEnv.
- Warning messages about data limitations are now printed when loading the package, creating a SpaceEnv, or creating a SpaceDataset.

### Changed
- Deleted `spacebench/datasets/` folder in favor of `spaceenv/env.py` more consie module.
- CausalDataset --> SpaceDataset
- SpaceEnv, SpaceDataset DataMaster are imported directly from spacebench in `spacebench/__init__.py`.
- Datamaster.list_datasets -> Datamaster.list_envs
- pip install will now install all dependencies automatically. It will not install dependencies of specific algorithms or examples.
- New optional `[all]` in pip install for the dependencies used in examples and algorithms.

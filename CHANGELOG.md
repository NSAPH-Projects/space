# spacebench (developing)


## Added

- DAPSm algorithm. Basic functionality with optimal and greedy matching.
- Evaluator classes with demo on evaluator_demo.ipynb
- SpaceEnv classes (same demo). Data automatically downloads given name.
- Added Masterfile but needs to be updated with new datasets.
- `list_envs` in DataMaster (previously `list_datasets`) now has an option for `binary=True` or `continuous=True`
- Added examples of benchmarks in the `examples/` folder.
- Add nice printing methods for SpaceDataset and SpaceEnv.
- Warning messages about data limitations are now printed when loading the package, creating a SpaceEnv, or creating a SpaceDataset.

## Changed
- Deleted `spacebench/datasets/` folder in favor of `spaceenv/env.py` more consie module.
- CausalDataset --> SpaceDataset
- SpaceEnv, SpaceDataset DataMaster are imported directly from spacebench in `spacebench/__init__.py`.
- Datamaster.list_datasets -> Datamaster.list_envs
- pip install will now install all dependencies automatically. It will not install dependencies of specific algorithms or examples.
- New optional `[all]` in pip install for the dependencies used in examples and algorithms.
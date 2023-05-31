# spacebench (developing)


## Added

- DAPSm algorithm. Basic functionality with optimal and greedy matching.
- Evaluator classes with demo on evaluator_demo.ipynb
- SpaceEnv classes (same demo). Data automatically downloads given name.
- Added Masterfile but needs to be updated with new datasets.

## Changed
- Deleted `spacebench/datasets/` folder in favor of `spaceenv/env.py` more consie module.
- CausalDataset --> SpaceDataset
- SpaceEnv, SpaceDataset DataMaster are imported directly from spacebench in `spacebench/__init__.py`.
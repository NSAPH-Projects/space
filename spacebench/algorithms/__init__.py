from spacebench.algorithms.base import SpaceAlgo
from spacebench.algorithms.dapsm import DAPSm
from spacebench.algorithms.nonspatial import OLS, XGBoost
from spacebench.algorithms.pysal_spreg import GMError, GMLag
from spacebench.algorithms.gcn import GCN
from spacebench.algorithms.spatialplus import SpatialPlus, Spatial
from spacebench.algorithms.deepspatialplus import (
    MLPSpatialPlus,
    MLPSpatial,
    DragonSpatial,
    DrnetSpatial,
)

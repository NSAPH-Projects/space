from spacebench.env import SpaceEnv, SpaceDataset
from spacebench.eval import DatasetEvaluator, EnvEvaluator
from spacebench.datamaster import DataMaster
import spacebench.algorithms.datautils


def _warn_user():
    if not getattr(_warn_user, "has_warned", False):
        warning_msg = (
            "WARNING ⚠️ : this package contains data with synthetic outcomes!\n"
            + "No inferences about the source data collection can be made.\n"
            + "By using it, you agree to understand its limitations and purpose.\n"
            + "The sole objective of SpaCE is to support the development of new\n"
            + "spatial confounding methods.\n"
        )
        print(warning_msg)
        _warn_user.has_warned = True


_warn_user()

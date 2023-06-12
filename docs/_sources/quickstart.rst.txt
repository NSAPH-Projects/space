Quickstart
==========

Available datasets 
------------------

Users can view the list of available datasets by executing the command below:

.. code-block:: python

    from spacebench.datasets import Datasets
    dm = Datasets()
    dm.list_envs()


Please be aware that these are only the names of the datasets. 


Semi-synthetic data environment
-------------------------------
To access the actual data, you must establish a semi-synthetic data environment. Let's say we want to download `healthd_dmgrcs_mortality_disc` which is the name of one of the datasets of the Air Pollution and Mortality collections. We can accomplish this by running the following command:

.. code-block:: python

    env = SpaceEnv('healthd_dmgrcs_mortality_disc', dir = "downloads")


The `downloads/healthd_dmgrcs_mortality_disc` directory will be created and the
data will be downloaded into it. Each dataset includes:

- training data
- true counterfactuals
- spatial graph with coordiates
- smoothness and confounding scores

.. Note::
    If the data is already downloaded, the `SpaceEnv` object will not download 
    it again.






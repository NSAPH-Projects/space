""" Command-line client """

import argparse
import dataverse as dv

parser = argparse.ArgumentParser()

parser.add_argument("predictor", choices=['xgboost', 'nn'], 
                    help="The model predictor algorithms")
parser.add_argument("datatype", choices=['binary', 'continuous'], 
                    help='Type of data: binary or continuous')
parser.add_argument("seed", type=int, help='Random seed')
parser.add_argument('--output_path', type=str, help='Path to save the data file')


args = parser.parse_args()


import spacebench.datasets.error_sampler as err
from spacebench.datasets.datasets import DatasetGenerator

# set random seed
err.set_random_seed(args.seed)

# download data
dv.download_data()
metadata_path, dataset_path = dv.get_dataset_metadata_and_path(
    args.predictor, args.datatype, args.output_path)

generator = DatasetGenerator.from_json(metadata_path)
dataset = generator.make_dataset()

dataset.save_dataset(dataset_path)
print("Dataset sampling completed.")

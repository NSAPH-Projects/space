""" Command-line client """

import os
import argparse
from spacebench.api.dataverse import DataverseAPI
from spacebench.datasets.error_sampler import GPSampler
from spacebench.datasets.datasets import DatasetGenerator
from pathlib import Path


class SpacebenchClient:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Spacebench client')
        self._configure_parser()

    def _configure_parser(self):
        self.parser.add_argument("predictor", choices=['xgboost', 'nn'],
                                 help="The model predictor algorithms")
        self.parser.add_argument("datatype", choices=['binary', 'continuous'],
                                 help='Type of data: binary or continuous')
        self.parser.add_argument("seed", type=int, help='Random seed')
        self.parser.add_argument('--remove_temp_files', action='store_true', default=False,
                    help='Remove temporary files (default: False)')
        self.parser.add_argument('--output_path', type=str, help='Path to save the data file')


    def download_and_generate_data(self, dvapi, predictor, datatype, output_path):
        dvapi.download_data(predictor, datatype)

        generator = DatasetGenerator.from_json(
             dvapi.core_data_loc+".json")
        dataset = generator.make_dataset()

        dataset.save_dataset(os.path.join(
            output_path, dvapi.data_filename+".csv"))
        print("Dataset sampling completed.")


    def run(self):
        args = self.parser.parse_args()

        target_dir = Path(args.output_path)
        if not target_dir.exists():
            print("The target directory doesn't exist")
            raise SystemExit(1)

        # download data
        dvapi = DataverseAPI()
        self.download_and_generate_data(
            dvapi, 
            args.predictor, 
            args.datatype, 
            args.output_path
        )
        if args.remove_temp_files:
            dvapi.remove_temp_files()


def main():
    client = SpacebenchClient()
    client.run()
    

if __name__ == "__main__":
    main()
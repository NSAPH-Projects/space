""" Command-line client """

import os
import logging
import argparse
from spacebench.api.dataverse import DataverseAPI
from spacebench.datasets.error_sampler import GPSampler
from spacebench.datasets.datasets import DatasetGenerator
from pathlib import Path


class SpacebenchClient:
    def __init__(self):
        self.dvapi = DataverseAPI()
        self.parser = argparse.ArgumentParser(
            description='A spacebench client for managing data files.')
        self.parser.add_argument(
            '--verbose', '-v', action='count', default=0, 
            help='Increase verbosity level (e.g., -vv for debug level)')
        self.subparsers = self.parser.add_subparsers(
            dest='command', required=True, help='Subcommands')
        
        self._configure_upload_parser()
        self._configure_list_parser()
        self._configure_download_parser()


    def _configure_upload_parser(self):
        parser_upload = self.subparsers.add_parser(
            'upload', 
            help='Upload a new data file to the collection.')
        parser_upload.add_argument('file', type=str, 
                                   help='Path to the data file to upload')
        parser_upload.set_defaults(func=self.upload)


    def _configure_list_parser(self):
        parser_list = self.subparsers.add_parser(
            'list', help='List existing data files from the collection.'
            )
        parser_list.add_argument('--include_fileid', action='store_true', 
                                     default=False, 
                                     help='List data files with file ID.')
        parser_list.set_defaults(func=self.list_data_files)


    def _configure_download_parser(self):
        # Download command
        parser_download = self.subparsers.add_parser('download', help='Download a data file')
        #parser_download.add_argument('file_id', type=str, help='ID of the data file to download')
        parser_download.set_defaults(func=self.download_and_generate_data)
        parser_download.add_argument("predictor", choices=['xgboost', 'nn'],
                                 help="The model predictor algorithms")
        parser_download.add_argument("datatype", choices=['binary', 'continuous'],
                                 help='Type of data: binary or continuous')
        parser_download.add_argument("seed", type=int, help='Random seed')
        parser_download.add_argument('--remove_temp_files', action='store_true', 
                                     default=False, 
                                     help='Remove temporary files (default: False)')
        parser_download.add_argument('--output_path', type=str, 
                                 help='Path to save the data file')

    def upload(self, args):
        # logging.info(f"Uploading file: {args.file}")
        # Implement the logic for uploading the file
        pass

    def list_data_files(self, args):
        logging.info("Existing data files:")
        print(self.dvapi.list_data_files(args.include_fileid))

    def _remove_temp_files(self):
        self.dvapi.remove_temp_files()

    def download_and_generate_data(self, args):
        target_dir = Path(args.output_path)
        if not target_dir.exists():
            print("The target directory doesn't exist")
            raise SystemExit(1)
        
        # download data
        logging.info(f"Downloading file with ID: {args.file_id}")
        self.dvapi.download_data(args.predictor, args.datatype)

        generator = DatasetGenerator.from_json(
             self.dvapi.core_data_loc+".json")
        dataset = generator.make_dataset()

        dataset.save_dataset(os.path.join(
            args.output_path, self.dvapi.data_filename+".csv"))
        logging.info(f"Dataset sampling completed.")

        if args.remove_temp_files:
            self._remove_temp_files()


    def run(self, args=None):
        args = self.parser.parse_args()
        loglevel = max(3 - args.verbose, 0) * 10
        logging.basicConfig(level=loglevel)
        args.func(args)


def main():
    client = SpacebenchClient()
    client.run()
    

if __name__ == "__main__":
    main()
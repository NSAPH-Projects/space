""" Command-line client """

import os
import argparse

from pathlib import Path
from spacebench.log import LOGGER
from spacebench.api.dataverse import DataverseAPI
from spacebench.datasets.error_sampler import GPSampler
from spacebench.datasets.datasets import DatasetGenerator


class SpacebenchClient:
    """
    A class representing the Spacebench Client.
    """

    def __init__(self):
        self.dvapi = DataverseAPI()
        self.parser = argparse.ArgumentParser(
            description="A spacebench client for managing data files.")
        self.parser.add_argument(
            "--verbose", "-v", action="count", default=0, 
            help="Increase verbosity level (e.g., -vv for debug level)")
        self.subparsers = self.parser.add_subparsers(
            dest="command", required=True, help="Subcommands")
        self._configure_upload_parser()
        self._configure_list_parser()
        self._configure_download_parser()

    def _configure_upload_parser(self):
        """
        Configures spacebench upload subparser.
        """
        parser_upload = self.subparsers.add_parser(
            "upload", 
            help="Upload a new data file to the collection.")
        parser_upload.add_argument(
            "--token", help="Set the authentication token")
        upload_subparsers = parser_upload.add_subparsers(
            dest="upload_command", required=True, help="Upload subcommands")

        # Add new datafile 
        parser_new = upload_subparsers.add_parser(
            "add", help="Upload a new data file")
        parser_new.add_argument(
            "file", type=str, help="Path to the data file to upload")
        parser_new.add_argument(
            "description", type=str, help="Data file description")
        parser_new.set_defaults(func=self.upload)

        # Replace existing file
        parser_replace = upload_subparsers.add_parser(
            "replace", help="Replace an existing data file")
        parser_replace.add_argument(
            "file_id", type=str, help="ID of the data file to replace")
        parser_replace.add_argument(
            "file", type=str, help="Path to the new data file")
        parser_replace.set_defaults(func=self.replace)

        # Finalize
        parser_finalize = upload_subparsers.add_parser(
            "publish", help="Finalize upload and publish updated dataset")
        parser_finalize.set_defaults(func=self.publish_uploaded_dataset)

    def _configure_list_parser(self):
        """
        Configures spacebench list subparser.
        """
        parser_list = self.subparsers.add_parser(
            "list", help="List existing data files from the collection")
        parser_list.add_argument(
            "--include_fileid", action="store_true", default=False, 
            help="List data files with file ID")
        parser_list.set_defaults(func=self.list_data_files)

    def _configure_download_parser(self):
        """
        Configures spacebench download subparser.
        """
        parser_download = self.subparsers.add_parser(
            "download", help="Download a data file")
        parser_download.set_defaults(
            func=self.download_and_generate_data)
        parser_download.add_argument(
            "predictor", choices=["xgboost", "nn"],
            help="The model predictor algorithms")
        parser_download.add_argument(
            "datatype", choices=["binary", "continuous"],
            help="Type of data: binary or continuous")
        parser_download.add_argument(
            "seed", type=int, help="Random seed")
        parser_download.add_argument(
            "--remove_temp_files", action="store_true", default=False,
            help="Remove temporary files (default: False)")
        parser_download.add_argument(
            "--output_path", type=str, help="Path to save the data file")

    def _remove_temp_files(self):
        """ Removes downloaded files from the temporary folder. """
        self.dvapi.remove_temp_files()

    def upload(self, args):
        """ Uploads data to the data collection. """
        LOGGER.info(f"Uploading file: {args.file}")
        self.dvapi.upload_data(
            args.file, args.description, args.token)

    def replace(self, args):
        """ Replaces a data file within the collection. """
        LOGGER.info(
            f"Replacing datafile with ID {args.file_id} with file: {args.file}")
        self.dvapi.replace(args.file, args.file_id, args.token)

    def publish_uploaded_dataset(self, args):
        """ Publishes updated dataset collection. """
        self.dvapi.publish_dataset(args.token)

    def list_data_files(self, args):
        """ Lists all existing data files. """
        LOGGER.info("Existing data files:")
        LOGGER.info(self.dvapi.list_data_files(args.include_fileid))

    def download_and_generate_data(self, args):
        """ Downloads data core and generates samples. """
        target_dir = Path(args.output_path)
        if not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
        
        # download data
        LOGGER.info(f"Starting the download.")
        self.dvapi.download_data(args.predictor, args.datatype)

        generator = DatasetGenerator.from_json(
             self.dvapi.core_data_loc+".json")
        dataset = generator.make_dataset()

        dataset.save_dataset(os.path.join(
            args.output_path, self.dvapi.data_filename+".csv"))
        LOGGER.info(f"Dataset sampling completed.")

        if args.remove_temp_files:
            self._remove_temp_files()

    def run(self, args=None):
        args = self.parser.parse_args()
        args.func(args)


def main():
    client = SpacebenchClient()
    client.run()
    

if __name__ == "__main__":
    main()
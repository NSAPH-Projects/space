"""
API for uploading and downloading data from Dataverse.
"""
import os
import json
import tempfile

from pyDataverse.models import Datafile
from pyDataverse.api import NativeApi, DataAccessApi
from spacebench.log import LOGGER


class DataverseAPI:
    """
    A class representing a Dataverse data repository API.
    """

    def __init__(self, dir: str | None = None):
        self.base_url = "https://dataverse.harvard.edu/"
        self.pid = "doi:10.7910/DVN/SYNPBS"
        self.api = NativeApi(self.base_url)
        self.data_api = DataAccessApi(self.base_url)
        self.dataset = self.__get_dataset()
        self.dir = dir
        if dir is None:
            self.dir = tempfile.gettempdir()
        self.files = self.__get_fileids_from_filenames()
        self.data_filename = ""

    @property
    def core_data_loc(self):
        """Returns core data location."""
        return os.path.join(self.dir, self.data_filename)

    def __get_dataset(self):
        return self.api.get_dataset(self.pid)

    def __get_fileids_from_filenames(self) -> dict:
        """Get fileid from filename for download."""
        files = {}

        files_list = self.dataset.json()["data"]["latestVersion"]["files"]

        for file in files_list:
            filename = file["dataFile"]["filename"]
            files[filename] = file["dataFile"]["id"]

        return files

    def list_data_files(self, include_fileid=False):
        """
        Lists data files from Dataverse.

        Args:
            include_fileid (bool): Include the file ID.
        """

        files_list = self.dataset.json()["data"]["latestVersion"]["files"]
        result = []
        for file in files_list:
            file_name = file["dataFile"]["filename"]
            file_desc = ""
            try:
                file_desc = file["dataFile"]["description"]
            except KeyError:
                pass
            file_id = file["dataFile"]["id"]
            if include_fileid:
                result.append(f"{file_name}\t{file_desc}\t{file_id}")
            else:
                result.append(f"{file_name}\t{file_desc}")

        return "\n".join(result)

    def remove_temp_files(self):
        """Removes temporary files."""

        for filename in self.files:
            file_path = os.path.join(self.dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                LOGGER.info(f"{filename} removed from the temporary directory.")

    def download_data(self, name: str) -> str:
        """Downloads core data and dicts from Dataverse."""

        # Download core data
        fileid = self.files[name]
        filename_temp_path = os.path.join(self.dir, name)

        if os.path.exists(filename_temp_path):
            LOGGER.info(f"File {name} already exists in the temporary directory.")
        else:
            response = self.data_api.get_datafile(fileid)

            if not os.path.exists(self.dir):
                os.makedirs(self.dir)

            with open(filename_temp_path, mode="wb") as temp_file:
                temp_file.write(response.content)

            LOGGER.info(
                f"Downloaded: filename {name}, id {fileid}, saved to {filename_temp_path}"
            )

        return filename_temp_path

    def publish_dataset(self, token):
        """
        Publish new dataset.

        Args:
            token (str): Dataverse API Token.
        """
        api = NativeApi(self.base_url, token)
        resp = api.publish_dataset(self.pid, release_type="major")
        if resp.json()["status"] == "OK":
            LOGGER.info("Dataset published.")

    def upload_data(self, file_path, description, token):
        """
        Upload data to the collection.

        Args:
            file_path (str): Filename
            description (str): Data file description.
            token (str): Dataverse API Token.
        """
        api = NativeApi(self.base_url, token)
        filename = os.path.basename(file_path)

        dv_datafile = Datafile()
        dv_datafile.set(
            {
                "pid": self.pid,
                "filename": filename,
                "description": description,
            }
        )
        LOGGER.info("File basename: " + filename)
        resp = api.upload_datafile(self.pid, file_path, dv_datafile.json())
        if resp.json()["status"] == "OK":
            LOGGER.info("Dataset uploaded.")

    def replace(self, file, file_id, token):
        """
        Replaces file in the collection.
        """
        api = NativeApi(self.base_url, token)
        filename = os.path.basename(file)

        dv_files_list = self.dataset.json()["data"]["latestVersion"]["files"]

        # keep the description from previous file version
        description = ""
        for dvf in dv_files_list:
            dv_file_id = dvf["dataFile"]["id"]
            if str(dv_file_id) == file_id:
                description = ""
                try:
                    description = dvf["dataFile"]["description"]
                except KeyError:
                    pass
                break

        json_dict = {
            "description": description,
            "forceReplace": True,
            "filename": filename,
            "label": filename,
        }

        json_str = json.dumps(json_dict)
        resp = api.replace_datafile(file_id, file, json_str, is_filepid=False)

        if resp.json()["status"] == "ERROR":
            LOGGER.error(f"An error at replacing the file: {resp.content}")

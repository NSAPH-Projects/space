"""
API for uploading and downloading data from Dataverse.
"""
import os
from pyDataverse.api import NativeApi, DataAccessApi


class DataverseAPI:

    def __init__(self, 
                 base_url = 'https://dataverse.harvard.edu/',
                 DOI = 'doi:10.7910/DVN/SYNPBS'
                 ):
        
        self.files = {}
        self.api = NativeApi(base_url)
        self.data_api = DataAccessApi(base_url)
        self.dataset = self.api.get_dataset(DOI)
        self.file_path = ""
        self.data_core = ""
    
    @property
    def core_data_loc(self):
        """ Returns core data location. """
        return os.path.join(
            self.file_path, 
            self.data_core
            )
    

    def __get_filenames_from_arguments(self, predictor, type):
        """ Get filenames for download. """

        json_filename = "medisynth-{}-{}.json".format(
            predictor, type)
        csv_filename = "medisynth-{}-{}.csv".format(
            predictor, type)
        
        self.data_core = json_filename
        
        filename_list = [
            "counties.geojson", 
            "counties.graphml",
            json_filename,
            csv_filename
            ]
        
        self.files = {
            key: None for key in filename_list
            }


    def __get_fileids_from_filenames(self):
        """ Get fileid from filename for download. """

        files_list = self.dataset.json()[
            'data']['latestVersion']['files']

        for file in files_list:
            filename = file["dataFile"]["filename"]

            if filename in self.files.keys():
                self.files[filename] = file["dataFile"]["id"]


    def download_data(self, predictor, type, output_path):
        """Downloads core data and dicts from Dataverse"""
    
        self.__get_filenames_from_arguments(
            predictor, type)
        self.__get_fileids_from_filenames()

        if output_path:
            self.file_path = output_path

        for filename in self.files:
            response = self.data_api.get_datafile(
                self.files[filename])

            with open(
                os.path.join(self.file_path, filename), 
                "wb") as f:
                f.write(response.content)
            
            print("File name {}, id {}".format(
                filename, self.files[filename]))


    def upload_data(self):
        """
        Upload data to the collection.
        """
        pass


"""
API for uploading and downloading data from Dataverse.
"""
import os
import tempfile
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
        self.temp_dir = tempfile.gettempdir()
        self.data_filename = ""
    
    @property
    def core_data_loc(self):
        """ Returns core data location. """
        return os.path.join(
            self.temp_dir, 
            self.data_filename
            )


    def __get_filenames_from_arguments(self, predictor, type):
        """ Get filenames for download. """

        self.data_filename = "medisynth-{}-{}".format(
            predictor, type)

        json_filename = self.data_filename + ".json"
        csv_filename = self.data_filename + ".csv"
        
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


    def list_data_files(self, include_fileid = False):
        """ Lists data files from Dataverse. """

        files_list = self.dataset.json()[
            'data']['latestVersion']['files']
        result = []
        for file in files_list:
            file_name = file["dataFile"]["filename"]
            file_desc = "" #file["dataFile"]["description"]
            file_id = file["dataFile"]["id"]
            if include_fileid:
                result.append(f"{file_name}\t{file_desc}\t{file_id}")
            else:
                result.append(f"{file_name}\t{file_desc}")
                

        return "\n".join(result)


    def remove_temp_files(self):
        """ Removes temporary files. """

        for filename in self.files:
            file_path = os.path.join(
                self.temp_dir, filename
            )
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(
                    f'{filename} removed from the temporary directory.')


    def download_data(self, predictor, type):
        """ Downloads core data and dicts from Dataverse. """
    
        self.__get_filenames_from_arguments(
            predictor, type)
        self.__get_fileids_from_filenames()

        for filename in self.files:
            filename_temp_path = os.path.join(
                self.temp_dir, filename
            )

            if os.path.exists(filename_temp_path):
                print(
                    f'File {filename} already exists in the temporary directory.')
            else:
                response = self.data_api.get_datafile(
                    self.files[filename])

                with open(filename_temp_path, mode='wb') as temp_file:
                    temp_file.write(response.content)
                
                print("Downloaded: file name {}, id {}".format(
                    filename, self.files[filename]))


    def upload_data(self):
        """
        Upload data to the collection.
        """
        # df = Datafile()
        # df.set({
        #     "pid" : args.doi,
        #     "filename" : f,
        #     "directoryLabel": root[5:],
        #     "description" : \
        #         "Uploaded with GitHub Action from {}.".format(
        #         args.repo),
        #     })
        # resp = api.upload_datafile(
        #     args.doi, join(root,f), df.json())
        pass


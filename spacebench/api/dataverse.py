"""
API for uploading and downloading data from Dataverse.
"""

from pyDataverse.api import NativeApi, DataAccessApi


def download_data():
    """Downloads core data and dicts from Dataverse"""
    base_url = 'https://dataverse.harvard.edu/'
    api = NativeApi(base_url)
    data_api = DataAccessApi(base_url)
    DOI = "doi:10.7910/DVN/SYNPBS"
    dataset = api.get_dataset(DOI)

    files_list = dataset.json()['data']['latestVersion']['files']

    for file in files_list:
        filename = file["dataFile"]["filename"]
        file_id = file["dataFile"]["id"]
        print("File name {}, id {}".format(filename, file_id))
        response = data_api.get_datafile(file_id)
        with open(filename, "wb") as f:
            f.write(response.content)


def get_dataset_metadata_and_path(predictor, binary, path):
    if "nn" in predictor:
        predictor_ = "nn"
    else:
        predictor_ = "xgboost"
    if "bin" in binary:
        binary_ = "binary"
    else:
        binary_ = "continuous"

    # get metadata path
    metadata_template = "medisynth-{}-{}.json"
    metadata_path = metadata_template.format(predictor_, binary_)
    dataset_path = "{}/medisynth-{}-{}-sample.csv".format(
        path, predictor_, binary_)
    return metadata_path, dataset_path

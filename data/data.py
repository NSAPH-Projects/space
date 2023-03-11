from pyDataverse.api import NativeApi, DataAccessApi

base_url = 'https://dataverse.harvard.edu/'

api = NativeApi(base_url)
data_api = DataAccessApi(base_url)

DOI = "doi:10.7910/DVN/SYNPBS"
dataset = api.get_dataset(DOI)

files_list = dataset.json()['data']['latestVersion']['files']
user_path = "../"

for file in files_list:
    filename = file["dataFile"]["filename"]
    filename = user_path + filename
    file_id = file["dataFile"]["id"]
    print("File name {}, id {}".format(filename, file_id))
    response = data_api.get_datafile(file_id)
    with open(filename, "wb") as f:
        f.write(response.content)

import subprocess
#process = subprocess.Popen(
#     ['python', '-c', 'from generate import generate_cf; generate_cf()'])

#process.wait()
print("Dataset generation completed.")

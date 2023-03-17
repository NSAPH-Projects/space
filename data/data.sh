#!/bin/bash

# run this script with
# curl -sSL https://raw.githubusercontent.com/NSAPH-Projects/space/master/data/data.sh | bash -s PREDICTOR BINARY|CONT SEED PATH

user_predictor="$1" # user provided scenario
user_binary="$2"
user_seed="$3" # user provided seed
user_path=$PWD # user provided path
if [ -n "$4" ]; then
    user_path="$4"
fi

echo "$user_predictor" && echo "$user_seed" && echo "$user_path"

mkdir .dataset_downloads
cd .dataset_downloads

curl -s https://raw.githubusercontent.com/NSAPH-Projects/space/dev/space/datasets.py -o datasets.py
curl -s https://raw.githubusercontent.com/NSAPH-Projects/space/dev/space/error_sampler.py -o error_sampler.py

python datasets.py $user_predictor $user_binary $user_seed $user_path &
mv counties.geojson $user_path/counties.geojson
mv counties.graphml $user_path/counties.graphml

PID=$! # get get_data.py PID
wait $PID

# Check the exit status of the Python script
if [ $? -eq 0 ]; then
    echo "The data is downloaded and generated using the seed $user_seed successfully."
    echo "The data can be found at $user_path."
else
    echo "Python script failed with exit code $?."
fi

cd .. ; rm -rf .dataset_downloads

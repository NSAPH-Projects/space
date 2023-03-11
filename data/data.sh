#!/bin/bash

# run this script with
# curl -sSL https://raw.githubusercontent.com/NSAPH-Projects/space/master/data/data.sh | bash

user_scenario="$1" # user provided scenario
user_seed="$2" # user provided seed
user_path=$PWD # user provided path
if [ -n "$3" ]; then
    user_path="$3"
fi

echo "$user_scenario" && echo "$user_seed" && echo "$user_path"
mkdir temp ; cd temp

curl -s https://raw.githubusercontent.com/NSAPH-Projects/space/master/data/requirements.txt -o requirements.txt
curl -s https://raw.githubusercontent.com/NSAPH-Projects/space/master/data/data.py -o data.py

# install requirements
mkdir env ; python3 -m venv env/
source env/bin/activate && pip install -r requirements.txt

python data.py $user_scenario $user_seed $user_path &
PID=$! # get get_data.py PID
wait $PID

# Check the exit status of the Python script
if [ $? -eq 0 ]; then
    echo "The data is downloaded and generated using the seed $user_seed successfully."
    echo "The data can be found at $user_path."
else
    echo "Python script failed with exit code $?."
fi

cd .. ; rm -rf temp

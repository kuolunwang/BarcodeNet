#!/usr/bin/env bash

echo "Install required libraries"
pip3 install virtualenv

virtualenv_name="BarcodeNet"
VIRTUALENV_FOLDER=$(pwd)/${virtualenv_name}
virtualenv ${virtualenv_name}

source ${VIRTUALENV_FOLDER}/bin/activate
python3 -m pip install -r requirements.txt
deactivate
echo "alias BarcodeNet='source ${VIRTUALENV_FOLDER}/bin/activate '" >> ~/.bashrc
source ~/.bashrc

# install carafe module
cd model/carafe && python3 setup.py develop
#!/usr/bin/env bash

sudo apt install python3.8-venv
python3 -m venv env
source env/bin/activate
pip install pip --upgrade

pip install -r requirements.txt

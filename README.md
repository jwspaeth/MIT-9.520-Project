# MIT-9.520-Project
Comparing convolutional weight sharing vs. locally connected network.

## Installation
- Create your base virtual environment of choice
- Run "pip install -r requirements.txt"

## Usage
There are two commands, "simple_train.py" and "simple_test.py". Each uses command-line arguments to access config files in the config/ folder. The syntax is "\<command\> -cd config/ -cn \<config name\>".  
To edit run parameters, edit their respective configuration files. For more info on hydra, see here: https://hydra.cc/docs/intro/

## Notes
If not using GPUs, make sure that in the config the entries trainer_cfg.precision = 32 and trainer_cfg.gpus = 0.
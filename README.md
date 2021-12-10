# MIT-9.520-Project
Comparing weight sharing and locality in convolutional, locally connected, and fully connected neural networks.

## Installation
- Create your base virtual environment of choice
- Run "pip install -r requirements.txt"

## Usage
- To download data, run the respective download_data() functions in src.datasets.\<file name\>.
- There are two commands, "simple_train.py" and "simple_test.py". Each uses command-line arguments to access config files in the config/ folder.
- The syntax is "\<command\> -cd config/ -cn \<config name\>".
- Example: "simple_train.py -cd config/ -cd train_cnn.yaml"
- To edit run parameters, edit their respective configuration files. For more info on hydra, see here: https://hydra.cc/docs/intro/

## Configs
Each experiment has its own training and test config, for a total of 20 configs.

## Notes
- If not using GPUs, make sure that in the config the entries trainer_cfg.precision = 32 and trainer_cfg.gpus = 0.

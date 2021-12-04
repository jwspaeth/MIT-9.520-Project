# MIT-9.520-Project
Comparing convolutional weight sharing vs. locally connected network.

## Installation
- Create your base virtual environment of choice
- Run "pip install -r requirements.txt"

## Usage
- To download data, run the respective download_data() functions in src.datasets.\<file name\>.
- There are two commands, "simple_train.py" and "simple_test.py". Each uses command-line arguments to access config files in the config/ folder.
- The syntax is "\<command\> -cd config/ -cn \<config name\>". Example: "simple_train.py -cd config/ -cd train_cnn.yaml"
- To edit run parameters, edit their respective configuration files. For more info on hydra, see here: https://hydra.cc/docs/intro/

## Configs
The two relevant configs are train_cnn.yaml and train_lcn.yaml, which correspond to training runs of the convolution and locally connected networks.

## Notes
- If not using GPUs, make sure that in the config the entries trainer_cfg.precision = 32 and trainer_cfg.gpus = 0.
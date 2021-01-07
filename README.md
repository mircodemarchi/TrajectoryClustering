# MLDL-Project

**Machine Learning** and **Deep Learning** courses project.

## Dataset

Download the dataset _zip_ file that you would like to test and put it in _data_ folder.
The following are the dataset explored in this project. 

- MotionSense: [onedrive/share/motion-sense.zip](https://univr-my.sharepoint.com/:u:/g/personal/mirco_demarchi_studenti_univr_it/Eab8vld0YLxNovlWBNrYiccBENiPIe53dVD_eJIYXqUc1g?e=o9IRFp)

- ScooterTrajectories: I'm sorry, this dataset is private, ask to the developer for the access.
    - [onedrive/share/scooter_trajectories.zip](): starting original dataset. I don't recommend to test the dataset on this, because it takes more or less 20 minutes to perform some merge and filtering operations.
    - [onedrive/share/scooter_trajectories_generated.zip](): resource of the dataset already filtered and merged. It weighs less and the analysis and training can start immediately.


## Quick run

1. Create the environment with all dependencies:
    - Conda: edit the `name` field of _environment.yml_ to change the environment name
    ```
    conda env create -f environment.yml
    ```
    - Virtualenv: not recommended because not tested
    ```
    # Create environment
    virtualenv <env-name>
    source <env-name>/bin/activate
    # Install requirements
    pip install -r requirements.txt 
    ```
2. Run the script with default configuration file _defconfig.ini_
```
python src/main.py
```

## Configurations

The project execution can be configured through a configuration file, in order to perform different operations on data. You can choose to perform only the data analysis or only the training with a selected technique, you can choose which dataset to test and how to filter and generate data, and so on. 

### Script parameters

```
usage: python src/main.py [-h] [--log LOG_LVL] [--config CONFIG_FILE]
optional arguments:
  -h, --help            show this help message and exit
  --log LOG_LVL, -l LOG_LVL
                        log level of the project: DEBUG, INFO, WARNING, ERROR, FATAL
  --config CONFIG_FILE, -c CONFIG_FILE
                        path to configuration file with all settings
```

- `--log` or `-l` argument: specify the log level to print. Every message with log level higher or equal to the log level specified will be printed. The log level severity is in the following order: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `FATAL`.

- `--config` or `-c` argument: specify the configuration file path (ex. _./config.ini_) to use for the script run. In the configuration file you can define the behavior of the project script on the dataset and it has a specific syntax that is the one specified in configparser python package ([configparser doc](https://docs.python.org/3/library/configparser.html)). See the [configuration file](#configuration-file) sub-section for more info.

### Configuration file

The configuration file (ex. _defconfig.ini_) is divided in different sections, on for each dataset that this project can perform. 

- MotionSense section

    This section start with `[MOTION-SENSE]` line and in the following lines you can specify the MotionSense test settings.

    - `skip`: bool
    - `save-file`: bool
    - `perform-analysis`: bool

- ScooterTrajectories section

    This section start with `[SCOOTER-TRAJECTORIES]` line and in the following lines you can specify the ScooterTrajectories test settings.

    - `skip`: bool
    - `generate-data`: bool
    - `chunk-size`: int
    - `max-chunk-num`: int optional
    - `perform-timestamp-clustering`: bool
    - `time-delta-clustering`: string
    - `load-generated`: bool
    - `perform-analysis`: bool
    - `num-of-pos-to-analyze`: int
    - `num-of-rental-in-dataset-to-analyze`: int


<br>
<p align="center">
    MDM <br>
    :monkey_face:
</p>

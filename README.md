# MLDL-Project

**Machine Learning** and **Deep Learning** courses project.

## Dataset

Download the dataset _zip_ file that you would like to test and put it in _data_ folder.
The following are the dataset studied in this project. 

- MotionSense: [onedrive/share/motion-sense.zip](https://univr-my.sharepoint.com/:u:/g/personal/mirco_demarchi_studenti_univr_it/Eab8vld0YLxNovlWBNrYiccBENiPIe53dVD_eJIYXqUc1g?e=o9IRFp)

- ScooterTrajectories: I'm sorry, this dataset is private, I hope you understand.
    - [onedrive/share/scooter_trajectories.zip](): starting original dataset. I don't recommend to perform your test on this, because it takes more or less 20 minutes to perform all merge and filter operations.
    - [onedrive/share/scooter_trajectories_generated.zip](): dataset already filtered and merged. It weighs less and the analysis and training can start immediately.


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
2. Run the script with default configuration
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

- `--log` or `-l` argument: specify the log level to print. Every message with log level higher or equal to the log level specified will be printed. The log level severity is in the following order: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `FATAL`. It this parameter is omitted the default value used is `DEBUG`.

- `--config` or `-c` argument: specify the configuration file path (ex. _./config.ini_) to use for test. In the configuration file you can define the behavior of the project script on the datasets. If this parameter is omitted the default configuration file taken is _\<proj-dir\>/defconfig.ini_. The syntax is the one specified in configparser python package ([configparser doc](https://docs.python.org/3/library/configparser.html)). See the [configuration file](#configuration-file) sub-section for more info.

### Configuration file

The configuration file (ex. _defconfig.ini_) is divided in different sections, on for each dataset that this project can perform. 

- MotionSense section

    This section start with `[MOTION-SENSE]` line and in the following lines you can specify the MotionSense test settings.

    - `skip`: bool
        
        Skip this section tests.

    - `save-file`: bool

        Save generated results in file. Image results are saved in _\<proj-dir\>/image_ folder, HTML results are saved in _\<proj-dir\>/html_ folder.

    - `perform-analysis`: bool

        Run dataset analysis and results analysis.


- ScooterTrajectories section

    This section start with `[SCOOTER-TRAJECTORIES]` line and in the following lines you can specify the ScooterTrajectories test settings.

    - `skip`: bool
        
        Skip this section tests.

    - `load-original-data`: bool

        Generate a new dataset form the original dataset and save it in the _\<proj-dir\>/data/scooter_trajectories_generated_ folder. This operation takes about 30 minutes to perform.

    - `load-generated-data`: bool

        Load already filtered, merged and handled dataset placed in _\<proj-dir\>/data/scooter_trajectories_generated_ folder or in _\<proj-dir\>/data/scooter_trajectories_generated.zip_ file.

    - `chunk-size`: int

        Chunk size used to load the positions of original dataset, in order to be able to manage a huge amount of data. 

    - `max-chunk-num`: int optional

        The index of the original dataset last chunk to parse. This is a limit to speed up the load and filter of original dataset. If omitted, it will take every chunk.  

    - `rental-num-to-analyze`: int optional

        Number of rentals to analyze. This value is a limit used to speed up the analysis and perform it in a reduced amount of data. If omitted, all rentals will be analyzed.

    - `only-north`: bool

        Perform analysis only in the northern part of the dataset positions in which there are the most significant data.

    - `perform-heuristic`: bool

        Performs timedelta heuristic, spreaddelta heuristic, edgedelta heuristic and coorddelta heuristic on generated dataset and overwrite the generated dataset with the computed heuristic columns.

    - `group-on-timedelta`: bool

        Groups the trajectory by the timedelta heuristic division using _timedelta_id_, otherwise groups the trajectory by rentals using _rental_id_. This setting is used by spreaddelta heuristic, edgedelta heuristic, coorddelta heuristic and the performed analysis. 

    - `timedelta`: int optional

        The delta value that if greater than the difference in time of two positions, consider each other as different trajectories. 

    - `spreaddelta`: int optional

        The delta value that if lower than the difference in spread (occupied area) between trajectories, consider the trajectories part of the same group.

    - `edgedelta`: int optional 

        The delta value that if lower than the difference in edges (start and stop coordinates) between trajectories, consider the trajectories part of the same group.

    - `perform-clustering`: string

        Perform the following clustering algorithms on generated dataset positions: k-means, mean-shift, gaussian mixture, ward hierarchical and full hierarchical.

    - `n-clusters`: int optional

        Number of clusters in input of clustering algorithms that need it. If omitted, it runs some WCSS clustering tests for Elbow method.

    - `with-pca`: bool

        Enable PCA features extraction in preparation of clustering.

    - `with-standardization`: bool

        Enable standardization of features in preparation of clustering.

    - `with-normalization`: bool

        Enable normalization of features in preparation of clustering.

    - `perform-data-analysis`: bool

        Perform analysis on rentals and positions as scatter plots, line plots and distribution plots. The results are saved as images in _\<proj-dir\>/image_ folder.

    - `perform-heuristic-analysis`: bool

        Perform analysis of heuristic columns (if previously performed and saved in your generated data) as scatter plots, line plots and distribution plots. The results are saved as images in _\<proj-dir\>/image_ folder.

    - `perform-map`: bool

        Show in your browser geographical maps and 3D maps of dataset generated positions in relation to heuristic process (if performed and saved in your generated data) and clustering (if `perform-clustering` is `true`). The results are saved as HTML files in _\<proj-dir\>/html_ folder.


<br>
<p align="center">
    MDM <br>
    :monkey_face:
</p>

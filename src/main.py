"""MachineLearning and DeepLearning courses project.

Python source file used only as a script that executes all the project for MachineLearning and DeepLearning courses.
The project involves in the following dataset:
-


Attributes
----------
parser : ArgumentParser
    parser of the script arguments
args : object
    result of the arguments parametrized

Functions
---------
main()
    project entry point
"""

import os
import time
import argparse
import logging
import configparser

from dataset import MotionSenseDS
from test import ScooterTrajectoriesTest
from util import DataAnalysis
from util import Log

log = Log(__name__, enable_console=True, enable_file=False)

# Script arguments
parser = argparse.ArgumentParser(description="Machine Learning and Deep Learning curses project", epilog="MDM project")
parser.add_argument("--log",
                    "-l",
                    dest="log_lvl",
                    required=False,
                    help="log level of the project: DEBUG, INFO, WARNING, ERROR, FATAL",
                    default="DEBUG")
parser.add_argument("--config",
                    "-c",
                    dest="config_file",
                    required=False,
                    help="path to configuration file with all settings",
                    default="defconfig.ini")
args = parser.parse_args()


def motion_sense_test(config: configparser.SectionProxy, log_lvl):
    if config.getboolean("skip"):
        return

    ms = MotionSenseDS(log_lvl=log_lvl)
    # dataset, target = ms.load(np.full(MotionSenseDS.TRIALS_NUM, 1))
    dataset, target = ms.load_all()
    ms.print_stats()

    da = DataAnalysis(dataset, list(dataset.columns)[0:12], target, save_file=True)
    da.show_target_distribution()
    da.show_relations()
    da.show_correlation_matrix()


def scooter_trajectories_test(config: configparser.SectionProxy, log_lvl):
    if config.getboolean("skip"):
        return

    st_test = ScooterTrajectoriesTest(
        log_lvl=log_lvl,
        chunk_size=None if config["chunk-size"] is None else config.getint("chunk-size"),
        max_chunk_num=None if config["max-chunk-num"] is None else config.getint("max-chunk-num"),
        rental_num_to_analyze=None if config["rental-num-to-analyze"] is None else config.getint("rental-num-to-analyze"),
        timedelta=config["timedelta"],
        spreaddelta=None if config["spreaddelta"] is None else config.getfloat("spreaddelta"),
        edgedelta=None if config["edgedelta"] is None else config.getfloat("edgedelta"),
        group_on_timedelta=config.getboolean("group-on-timedelta"),
        n_clusters=None if config["n-clusters"] is None else config.getint("n-clusters"),
        with_pca=config.getboolean("with-pca"),
        with_standardization=config.getboolean("with-standardization"),
        with_normalization=config.getboolean("with-normalization"),
        only_north=config.getboolean("only-north"),
        exam=config.getboolean("exam")
    )

    if config.getboolean("load-original-data"):
        st_test.load_from_original()
        st_test.store()

    if config.getboolean("load-generated-data"):
        st_test.load_from_generated()

    if not st_test.is_data_processed():
        return

    # ML
    if config.getboolean("perform-heuristic"):
        st_test.heuristic()
        st_test.store()

    if config.getboolean("perform-clustering"):
        st_test.clustering()

    # Analysis
    st_test.stats()
    st_test.partition_stats()
    if config.getboolean("perform-data-analysis"):
        st_test.generated_data_analysis()

    if config.getboolean("perform-heuristic-analysis"):
        st_test.heuristic_data_analysis()

    if config.getboolean("perform-clustering") and st_test.is_clustering_processed():
        st_test.clusterized_data_analysis()
        st_test.cluster_maps()
        st_test.cluster_maps_3d()

    if config.getboolean("perform-map"):
        st_test.maps()


def main():
    # Handle args
    log_lvl = getattr(logging, args.log_lvl.upper(), logging.DEBUG)
    config_file = os.path.join(os.path.dirname(__file__), "..", args.config_file)
    if not os.path.exists(config_file):
        log.f("Configuration file do not find, exit")
        return

    # Handle config
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(config_file)

    main_tests = {
        "MotionSense": {
            "test": motion_sense_test,
            "config": config["MOTION-SENSE"]
        },
        "ScooterTrajectories": {
            "test": scooter_trajectories_test,
            "config": config["SCOOTER-TRAJECTORIES"]
        },
    }

    # Start test
    start = time.time()
    for test in main_tests:
        main_tests[test]["test"](main_tests[test]["config"], log_lvl)
    end = time.time()

    # Calculate time
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


if __name__ == '__main__':
    main()

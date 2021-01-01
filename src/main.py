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
import numpy as np

from util import MotionSenseDS, ScooterTrajectoriesDS
from util import DataAnalysis

# Script arguments
parser = argparse.ArgumentParser(description="Machine Learning and Deep Learning curses project", epilog="MDM project")
parser.add_argument("--log",
                    "-l",
                    dest="log_lvl",
                    required=False,
                    help="Configuration file with all server specification",
                    default="DEBUG")
parser.add_argument("--config",
                    "-c",
                    dest="config_file",
                    required=False,
                    help="Configuration file with all server specification",
                    default="config.ini")
args = parser.parse_args()


def motion_sense_test(config, log_lvl):
    ms = MotionSenseDS()
    # dataset, target = ms.load(np.full(MotionSenseDS.TRIALS_NUM, 1))
    dataset, target = ms.load_all()
    ms.print_stats()

    da = DataAnalysis(dataset, list(dataset.columns)[0:12], target)
    da.show_target_distribution(file=True)
    da.show_relations(file=True)
    da.show_correlation_matrix(file=True)


def scooter_trajectories(config, log_lvl):
    st = ScooterTrajectoriesDS().load_all(chunksize=100000).print_stats().to_csv()
    print(st.dataset.head())


def main():
    # Handle args
    log_lvl = getattr(logging, args.log_lvl.upper(), logging.DEBUG)
    config_file = os.path.join(os.path.dirname(__file__), args.config_file)

    # Handle config
    config = configparser.ConfigParser()
    config.read(config_file)

    main_tests = {
        # "MotionSense": motion_sense_test,
        "ScooterTrajectories": scooter_trajectories,
    }

    # Start test
    start = time.time()
    for test in main_tests:
        main_tests[test](config, log_lvl)
    end = time.time()

    # Calculate time
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


if __name__ == '__main__':
    main()

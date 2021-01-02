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

from util import MotionSenseDS, ScooterTrajectoriesDS
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

    da = DataAnalysis(dataset, list(dataset.columns)[0:12], target)
    da.show_target_distribution(file=True)
    da.show_relations(file=True)
    da.show_correlation_matrix(file=True)


def scooter_trajectories(config: configparser.SectionProxy, log_lvl):
    if config.getboolean("skip"):
        return

    loaded_data = False
    st = ScooterTrajectoriesDS(log_lvl=log_lvl)
    if config.getboolean("generate-data"):
        st.generate_all(chunksize=config["chunk-size"],
                        max_chunknum=None if config["max-chunk-num"] is None else config.getint("max-chunk-num"))
        st.to_csv()
        loaded_data = True

    if config.getboolean("load-generated"):
        st.load_generated()
        loaded_data = True

    if loaded_data:
        st.print_stats()

        if config["perform-analysis"]:
            DataAnalysis(st.rental, ScooterTrajectoriesDS.RENTAL_ANALYSIS_COLS).show_distribution()
            DataAnalysis(st.pos, ScooterTrajectoriesDS.POS_ANALYSIS_COLS).show_distribution()


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
            "test": scooter_trajectories,
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

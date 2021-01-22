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

from dataset import MotionSenseDS, ScooterTrajectoriesDS
from dataset.constant import MotionSenseC as MSC
from dataset.constant import ScooterTrajectoriesC as STC
from util import DataAnalysis
from util import Log

from ml import Clustering

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


def scooter_trajectories(config: configparser.SectionProxy, log_lvl):
    if config.getboolean("skip"):
        return

    st = ScooterTrajectoriesDS(log_lvl=log_lvl)
    if config.getboolean("load-original-data"):
        st.generate_all(chunksize=config.getint("chunk-size"),
                        max_chunknum=None if config["max-chunk-num"] is None else config.getint("max-chunk-num"))
        st.to_csv()

    if config.getboolean("load-generated-data"):
        st.load_generated()

    if st.dataset.empty:
        return

    # ML
    if config.getboolean("perform-timedelta-heuristic") or config.getboolean("perform-spreaddelta-heuristic")\
            or config.getboolean("perform-edgedelta-heuristic") or config.getboolean("perform-coorddelta-heuristic"):
        groupby = STC.POS_GEN_RENTAL_ID_CN
        if config.getboolean("perform-timedelta-heuristic"):
            st.timedelta_heuristic(timedelta=config["timedelta"])
            groupby = [STC.POS_GEN_RENTAL_ID_CN, STC.POS_GEN_TIMEDELTA_ID_CN]

        if config.getboolean("perform-spreaddelta-heuristic"):
            st.spreaddelta_heuristic(spreaddelta=config.getfloat("spreaddelta"), groupby=groupby)

        if config.getboolean("perform-edgedelta-heuristic"):
            st.edgedelta_heuristic(edgedelta=config.getfloat("edgedelta"), groupby=groupby)

        if config.getboolean("perform-coorddelta-heuristic"):
            st.coorddelta_heuristic(spreaddelta=config.getfloat("spreaddelta"), edgedelta=config.getfloat("edgedelta"),
                                    groupby=groupby)
        st.to_csv()

    if config.getboolean("perform-kmeans"):
        dataset_for_clustering, cols = st.prepare_for_clustering()
        #dataset_for_clustering = dataset_for_clustering.loc[dataset_for_clustering[STC.MERGE_POS_LATITUDE_CN] >= 44.0]
        kmeans = Clustering(dataset_for_clustering, STC.CLUSTERING_COLS, dataset_name="ScooterTrajectories")

        if config["n-clusters"] is None:
            kmeans.test(standardize=config.getboolean("with-standardization"), pca=config.getboolean("with-pca"))
            kmeans.show_wcss()
        else:
            kmeans.exec(config.getint("n-clusters"), standardize=config.getboolean("with-standardization"),
                        pca=config.getboolean("with-pca"))
            dataset_for_clustering["labels"] = kmeans.labels
            # All dataset k-means analysis
            da_dataset = DataAnalysis(dataset_for_clustering, cols, dataset_name="ScooterTrajectories",
                                      save_file=True, prefix="kmeans_all")
            da_dataset.show_joint(on=[STC.MERGE_POS_LONGITUDE_CN, STC.MERGE_POS_LATITUDE_CN, "labels"])

            rental_to_analyze = dataset_for_clustering[STC.MERGE_RENTAL_ID_CN].unique()[
                                :config.getint("num-of-rental-in-dataset-to-analyze")]
            dataset_to_analyze = dataset_for_clustering.loc[
                dataset_for_clustering[STC.MERGE_RENTAL_ID_CN].isin(rental_to_analyze)]
            # Bottom left dataset k-means analysis
            dataset_bl = dataset_to_analyze.loc[dataset_to_analyze[STC.MERGE_POS_LATITUDE_CN] < 44.0]
            da_dataset_bl = DataAnalysis(dataset_bl, dataset_bl.columns, dataset_name="ScooterTrajectories",
                                         save_file=True, prefix="kmeans_bottom_left")
            da_dataset_bl.show_joint(on=[STC.MERGE_POS_LONGITUDE_CN, STC.MERGE_POS_LATITUDE_CN, "labels"])
            # Top right dataset k-means analysis
            dataset_tr = dataset_to_analyze.loc[dataset_to_analyze[STC.MERGE_POS_LATITUDE_CN] >= 44.0]
            da_dataset_tr = DataAnalysis(dataset_tr, dataset_tr.columns, dataset_name="ScooterTrajectories",
                                         save_file=True, prefix="kmeans_top_right")
            da_dataset_tr.show_joint(on=[STC.MERGE_POS_LONGITUDE_CN, STC.MERGE_POS_LATITUDE_CN, "labels"])

    # Analysis
    st.print_stats()

    pos_to_analyze = st.pos if config["num-of-pos-to-analyze"] is None else st.pos.iloc[
                                                                            :config.getint("num-of-pos-to-analyze")]
    rental_to_analyze = st.pos[STC.POS_GEN_RENTAL_ID_CN].unique()[
                        :config.getint("num-of-rental-in-dataset-to-analyze")]
    dataset_to_analyze = st.pos.loc[st.pos[STC.POS_GEN_RENTAL_ID_CN].isin(rental_to_analyze)]
    dataset_bl = dataset_to_analyze.loc[dataset_to_analyze[STC.POS_GEN_LATITUDE_CN] < 44.0]
    dataset_tr = dataset_to_analyze.loc[dataset_to_analyze[STC.POS_GEN_LATITUDE_CN] >= 44.0]
    if config.getboolean("perform-data-analysis"):
        # Rental analysis
        da_rental = DataAnalysis(st.rental, STC.RENTAL_ANALYSIS_COLS,
                                 dataset_name="ScooterTrajectories",
                                 save_file=True, prefix="rental")
        da_rental.show_distributions().show_2d_distributions(STC.RENTAL_2D_ANALYSIS_COUPLES)
        for c in STC.RENTAL_2D_ANALYSIS_COUPLES:
            da_rental.show_joint(on=c)

        # Pos analysis
        da_pos = DataAnalysis(pos_to_analyze, STC.POS_GEN_ANALYSIS_COLS,
                              dataset_name="ScooterTrajectories", save_file=True, prefix="pos")
        da_pos.show_distributions().show_2d_distributions(STC.POS_GEN_2D_ANALYSIS_COUPLES)
        for c in STC.POS_GEN_2D_ANALYSIS_COUPLES:
            da_pos.show_joint(on=c)

        # Result analysis
        da_dataset = DataAnalysis(dataset_to_analyze, st.pos.columns, dataset_name="ScooterTrajectories",
                                  save_file=True, prefix="data_all")
        da_dataset.show_line(on=STC.POS_GEN_OVER_RENTAL_ANALYSIS_TUPLE)

        da_dataset_bl = DataAnalysis(dataset_bl, st.pos.columns, dataset_name="ScooterTrajectories",
                                     save_file=True, prefix="data_bottom_left")
        da_dataset_bl.show_line(on=STC.POS_GEN_OVER_RENTAL_ANALYSIS_TUPLE)
        da_dataset_bl.show_joint(on=STC.POS_GEN_OVER_RENTAL_ANALYSIS_TUPLE)

        da_dataset_tr = DataAnalysis(dataset_tr, st.pos.columns, dataset_name="ScooterTrajectories",
                                     save_file=True, prefix="data_top_right")
        da_dataset_tr.show_line(on=STC.POS_GEN_OVER_RENTAL_ANALYSIS_TUPLE)
        da_dataset_tr.show_joint(on=STC.POS_GEN_OVER_RENTAL_ANALYSIS_TUPLE)

    if config.getboolean("perform-heuristic-analysis"):
        da_dataset_bl = DataAnalysis(dataset_bl, st.pos.columns, dataset_name="ScooterTrajectories",
                                     save_file=True, prefix="heuristic_bottom_left")
        da_dataset_bl.show_line(on=STC.POS_GEN_OVER_TIMEDELTA_ANALYSIS_TUPLE)
        da_dataset_bl.show_joint(on=STC.POS_GEN_OVER_SPREADDELTA_ANALYSIS_TUPLE)
        da_dataset_bl.show_joint(on=STC.POS_GEN_OVER_EDGEDELTA_ANALYSIS_TUPLE)
        da_dataset_bl.show_joint(on=STC.POS_GEN_OVER_COORDDELTA_ANALYSIS_TUPLE)

        da_dataset_tr = DataAnalysis(dataset_tr, st.pos.columns, dataset_name="ScooterTrajectories",
                                     save_file=True, prefix="heuristic_top_right")
        da_dataset_tr.show_line(on=STC.POS_GEN_OVER_TIMEDELTA_ANALYSIS_TUPLE)
        da_dataset_tr.show_joint(on=STC.POS_GEN_OVER_SPREADDELTA_ANALYSIS_TUPLE)
        da_dataset_tr.show_joint(on=STC.POS_GEN_OVER_EDGEDELTA_ANALYSIS_TUPLE)
        da_dataset_tr.show_joint(on=STC.POS_GEN_OVER_COORDDELTA_ANALYSIS_TUPLE)

    if config.getboolean("perform-clustering-analysis"):
        pass

    if config.getboolean("perform-map"):
        rental_to_analyze = st.pos[STC.POS_GEN_RENTAL_ID_CN].unique()[
                            :config.getint("num-of-rental-in-dataset-to-analyze")]
        dataset_to_analyze = st.pos.loc[st.pos[STC.POS_GEN_RENTAL_ID_CN].isin(rental_to_analyze)]
        da_dataset = DataAnalysis(dataset_to_analyze, st.pos.columns, dataset_name="ScooterTrajectories",
                                  save_file=True)
        da_dataset.show_line_map(on=STC.POS_GEN_OVER_RENTAL_MAP_TUPLE,
                                 hover_data=STC.POS_GEN_OVER_RENTAL_MAP_HOVER_DATA)
        da_dataset.show_scatter_map(on=STC.POS_GEN_OVER_CLUSTER_MAP_TUPLE,
                                    hover_data=STC.POS_GEN_OVER_CLUSTER_MAP_HOVER_DATA)


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

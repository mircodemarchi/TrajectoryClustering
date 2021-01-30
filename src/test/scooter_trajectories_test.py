import time
import pandas as pd

from dataset import ScooterTrajectoriesDS
from dataset.constant import ScooterTrajectoriesC as STC

from ml import Clustering

from util.analysis import DataAnalysis
from util.log import Log
from util.util import get_elapsed

log = Log(__name__, enable_console=True, enable_file=False)

DATASET_NAME = "ScooterTrajectories"
SAVE_FILE = False

RENTAL_IMG_FN_PREFIX = "rental"
POS_IMG_FN_PREFIX = "pos"
POS_OVER_RENTAL_IMG_FN_PREFIX = "data_all"
CLUSTER_IMG_FN_PREFIX = "kmeans"
CLUSTER_TEST_IMG_FN_PREFIX = "kmeans_test"
HEURISTIC_IMG_FN_PREFIX = "heuristic"

FORCE_RENTAL_NUM_TO_ANALYZE = 100


class ScooterTrajectoriesTest:
    def __init__(self, log_lvl=None, chunk_size=None, max_chunk_num=None, rental_num_to_analyze=None,
                 timedelta=None, spreaddelta=None, edgedelta=None, group_on_timedelta=True,
                 n_clusters=None, with_pca=False, with_standardization=False, with_normalization=False,
                 only_north=False):
        self.st = ScooterTrajectoriesDS(log_lvl=log_lvl)
        # Generation settings
        self.chunk_size = chunk_size
        self.max_chunk_num = max_chunk_num
        # Analysis settings
        self.rental_num_to_analyze = rental_num_to_analyze
        # Heuristic settings
        self.timedelta = timedelta
        self.spreaddelta = spreaddelta
        self.edgedelta = edgedelta
        # Clustering settings
        self.n_clusters = n_clusters
        self.with_pca = with_pca
        self.with_standardization = with_standardization
        self.with_normalization = with_normalization
        self.only_north = only_north

        # Others
        self.clustering_done = False
        self.data_clusterized = None
        self.partitions_clusterized = None
        self.groupby = [STC.POS_GEN_RENTAL_ID_CN,
                        STC.POS_GEN_TIMEDELTA_ID_CN] if group_on_timedelta else STC.POS_GEN_RENTAL_ID_CN

    def __partition(self, dataset, only_north=False):
        dataset_south = dataset.loc[dataset[STC.MERGE_POS_LATITUDE_CN] < 43.0]
        dataset_north = dataset.loc[(dataset[STC.MERGE_POS_LATITUDE_CN] >= 43.0) &
                                    (dataset[STC.MERGE_POS_LATITUDE_CN] < 44.3)]

        dataset_north_east = dataset_north.loc[dataset_north[STC.MERGE_POS_LONGITUDE_CN] >= 12.0]
        dataset_north_west = dataset_north.loc[(dataset_north[STC.MERGE_POS_LONGITUDE_CN] < 10.0) &
                                               (dataset_north[STC.MERGE_POS_LONGITUDE_CN] > 8.0)]
        dataset_south_east = dataset_south.loc[(dataset_south[STC.MERGE_POS_LONGITUDE_CN] >= 13.0) &
                                               (dataset_south[STC.MERGE_POS_LONGITUDE_CN] < 14.2)]
        dataset_south_west = dataset_south.loc[(dataset_south[STC.MERGE_POS_LONGITUDE_CN] < 13.0) &
                                               (dataset_south[STC.MERGE_POS_LONGITUDE_CN] > 11.7)]

        data_partitioned = {"N-E": dataset_north_east, "N-W": dataset_north_west}
        if not only_north:
            data_partitioned["S-E"] = dataset_south_east
            data_partitioned["S-W"] = dataset_south_west

        return data_partitioned

    def __filter(self, dataset):
        if self.rental_num_to_analyze is None:
            return dataset
        else:
            rental_to_analyze = dataset[STC.MERGE_RENTAL_ID_CN].unique()[:self.rental_num_to_analyze]
            return dataset.loc[dataset[STC.MERGE_RENTAL_ID_CN].isin(rental_to_analyze)]

    def __prepare(self):
        log.d("Test {} prepare data for clustering".format(DATASET_NAME))
        start = time.time()
        merge_cols_pos_gen_map = dict(zip(STC.POS_GEN_COLS_MERGE_MAP.values(), STC.POS_GEN_COLS_MERGE_MAP.keys()))
        pos_sort_cols = STC.POS_GEN_SORT_COLS
        merge_sort_cols = [merge_cols_pos_gen_map[c] for c in pos_sort_cols]

        # Sort to align the frame rows
        ordered_merge = self.st.merge.sort_values(by=merge_sort_cols, ignore_index=True)
        join = ordered_merge
        if self.is_heuristic_processed():
            ordered_pos = self.st.pos.sort_values(by=pos_sort_cols, ignore_index=True)
            join = pd.concat([join, ordered_pos[STC.POS_GEN_HEURISTIC_COLS]], axis=1)

        # Convert time columns in float
        join_time_cols = STC.MERGE_TIME_COLS
        join[join_time_cols] = join[join_time_cols].applymap(lambda x: x.timestamp())

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))
        return join

    def __line_joint_analysis(self, dataset, joint_list=None, line_list=None, prefix=None):
        da = DataAnalysis(dataset, dataset.columns, dataset_name=DATASET_NAME, save_file=SAVE_FILE, prefix=prefix)
        if joint_list is not None:
            for t in joint_list:
                da.show_joint(on=t)

        if line_list is not None:
            for t in line_list:
                da.show_line(on=t, groupby=self.groupby)
        return self

    def __overview_analysis(self, dataset, cols, couples=None, prefix=None):
        da = DataAnalysis(dataset, cols, dataset_name=DATASET_NAME, save_file=SAVE_FILE, prefix=prefix)
        da.show_distributions()
        if couples is not None:
            da.show_2d_distributions(couples)
            for c in couples:
                da.show_joint(on=c)
        return self

    def __cardinal_analysis(self, dataset, joint_list=None, line_list=None, prefix=None, skip_filter=True):
        partitions = self.__partition(dataset, only_north=self.only_north)
        for key in partitions:
            prefix_with_cardinal = "{}_{}".format(prefix, key) if prefix is not None else key
            if not skip_filter:
                partitions[key] = self.__filter(partitions[key])
            self.__line_joint_analysis(partitions[key], joint_list, line_list, prefix_with_cardinal)
        return self

    def load_from_original(self):
        log.d("Test {} start load from original data".format(DATASET_NAME))
        if self.is_data_processed():
            log.w("Test {} load_from_original: already processed".format(DATASET_NAME))
            return self

        if self.chunk_size is not None and self.max_chunk_num is not None:
            self.st.generate_all(chunksize=self.chunk_size, max_chunknum=self.max_chunk_num)
        elif self.chunk_size is not None:
            self.st.generate_all(chunksize=self.chunk_size)
        elif self.max_chunk_num is not None:
            self.st.generate_all(max_chunknum=self.max_chunk_num)
        else:
            self.st.generate_all()
        self.load_done = True
        return self

    def load_from_generated(self):
        log.d("Test {} start load from already generated data".format(DATASET_NAME))
        if self.is_data_processed():
            log.w("Test {} load_from_generated: already processed".format(DATASET_NAME))
            return self

        self.st.load_generated()
        self.load_done = True
        return self

    def store(self):
        log.d("Test {} store data processed".format(DATASET_NAME))
        self.st.to_csv()
        return self

    def is_data_processed(self):
        return not self.st.empty()

    def is_heuristic_processed(self):
        return not self.st.heuristic_empty()

    def is_clustering_processed(self):
        return self.clustering_done and self.data_clusterized is not None and self.partitions_clusterized is not None

    def heuristic(self):
        log.d("Test {} start heuristic process".format(DATASET_NAME))
        if self.is_heuristic_processed():
            log.w("Test {} heuristic: columns already exist, overwriting them".format(DATASET_NAME))

        if not self.is_data_processed():
            log.e("Test {} heuristic: you have to process data earlier".format(DATASET_NAME))
            return self
        self.st.timedelta_heuristic(timedelta=self.timedelta)
        self.st.spreaddelta_heuristic(spreaddelta=self.spreaddelta, groupby=self.groupby)
        self.st.edgedelta_heuristic(edgedelta=self.edgedelta, groupby=self.groupby)
        self.st.coorddelta_heuristic(spreaddelta=self.spreaddelta, edgedelta=self.edgedelta, groupby=self.groupby)
        return self

    def test_clustering(self):
        prefix = CLUSTER_TEST_IMG_FN_PREFIX
        log.d("Test {} start clustering wcss test".format(DATASET_NAME))
        if not self.is_heuristic_processed():
            log.e("Test {} test_clustering: you have to process heuristic earlier".format(DATASET_NAME))
            return self
        dataset_for_clustering = self.__prepare()
        partitions = self.__partition(dataset_for_clustering, only_north=self.only_north)

        # Perform clustering tests
        kmeans = Clustering(dataset_for_clustering, STC.CLUSTERING_COLS, dataset_name="ScooterTrajectories")
        kmeans.test(range_clusters=range(1, 20), standardize=self.with_standardization,
                    normalize=self.with_normalization,  pca=self.with_pca, components=STC.CLUSTERING_COMPONENTS)
        kmeans.show_wcss(save_file=SAVE_FILE, prefix=prefix + "all")
        for key in partitions:
            kmeans = Clustering(partitions[key], STC.CLUSTERING_COLS, dataset_name="ScooterTrajectories")
            kmeans.test(range_clusters=range(1, 30), standardize=self.with_standardization,
                        normalize=self.with_normalization, pca=self.with_pca, components=STC.CLUSTERING_COMPONENTS)
            kmeans.show_wcss(save_file=SAVE_FILE, prefix=prefix + key)

    def clustering(self):
        if self.n_clusters is None:
            self.test_clustering()
            return self

        log.d("Test {} start clustering k-means analysis".format(DATASET_NAME))
        if not self.is_heuristic_processed():
            log.e("Test {} clustering: you have to process heuristic earlier".format(DATASET_NAME))
            return self

        components = STC.CLUSTERING_COMPONENTS  # STC.CLUSTERING_COMPONENTS or None or a number
        dataset_for_clustering = self.__prepare()

        # Perform clustering in relation to the entire data
        kmeans = Clustering(dataset_for_clustering, STC.CLUSTERING_COLS, dataset_name="ScooterTrajectories")
        kmeans.exec(self.n_clusters, standardize=self.with_standardization, normalize=self.with_normalization,
                    pca=self.with_pca, components=components)
        dataset_for_clustering[STC.CLUSTER_ID_CN] = kmeans.labels

        # Perform clustering in relation to each partition
        partitions = self.__partition(dataset_for_clustering, only_north=self.only_north)
        for key in partitions:
            kmeans = Clustering(partitions[key], STC.CLUSTERING_COLS, dataset_name="ScooterTrajectories")
            kmeans.exec(self.n_clusters, standardize=self.with_standardization, normalize=self.with_normalization,
                        pca=self.with_pca, components=components)
            partitions[key][STC.CLUSTER_ID_CN] = kmeans.labels

        self.data_clusterized = dataset_for_clustering
        self.partitions_clusterized = partitions
        self.clustering_done = True

    def generated_data_analysis(self):
        log.d("Test {} generated data analysis".format(DATASET_NAME))
        if not self.is_data_processed():
            log.e("Test {} generated data analysis: you have to process data earlier".format(DATASET_NAME))
            return self

        # Rental analysis
        rental_to_analyze = self.__rental_filter(self.st.rental)
        if self.rental_num_to_analyze is not None:
            rental_to_analyze = rental_to_analyze.iloc[:self.rental_num_to_analyze]
        self.__overview_analysis(rental_to_analyze, STC.RENTAL_ANALYSIS_COLS, STC.RENTAL_2D_ANALYSIS_COUPLES,
                                 RENTAL_IMG_FN_PREFIX)
        # Pos analysis
        pos_to_analyze = self.__filter(self.__pos_filter(self.st.pos))
        self.__overview_analysis(pos_to_analyze, STC.POS_GEN_ANALYSIS_COLS, STC.POS_GEN_2D_ANALYSIS_COUPLES,
                                 POS_IMG_FN_PREFIX)
        self.__cardinal_analysis(self.st.pos, prefix=POS_IMG_FN_PREFIX,
                                 line_list=[STC.POS_GEN_2D_ANALYSIS_COUPLES[0]],
                                 joint_list=[STC.POS_GEN_2D_ANALYSIS_COUPLES[0]])
        # Pos over Rental analysis
        self.__line_joint_analysis(pos_to_analyze,
                                   line_list=[STC.POS_GEN_OVER_RENTAL_ANALYSIS_TUPLE],
                                   prefix=POS_OVER_RENTAL_IMG_FN_PREFIX)
        self.__cardinal_analysis(self.st.pos, prefix=POS_OVER_RENTAL_IMG_FN_PREFIX,
                                 line_list=[STC.POS_GEN_OVER_RENTAL_ANALYSIS_TUPLE],
                                 joint_list=[STC.POS_GEN_OVER_RENTAL_ANALYSIS_TUPLE],
                                 line_3d_list=[STC.POS_GEN_OVER_RENTAL_ANALYSIS_TUPLE])
        return self

    def heuristic_data_analysis(self):
        log.d("Test {} heuristic data analysis".format(DATASET_NAME))
        if not self.is_heuristic_processed():
            log.e("Test {} heuristic data analysis: you have to process heuristic earlier".format(DATASET_NAME))
            return self
        p = HEURISTIC_IMG_FN_PREFIX
        pos_filter = self.__pos_filter(self.st.pos)
        pos_to_analyze = self.__filter(pos_filter)
        self.__overview_analysis(pos_to_analyze, STC.POS_GEN_HEURISTIC_ID_COLS, prefix=p + "_id")
        self.__overview_analysis(pos_to_analyze, STC.POS_GEN_HEURISTIC_INFO_COLS, prefix=p + "_info")
        self.__cardinal_analysis(pos_filter, prefix=p,
                                 line_list=[STC.POS_GEN_OVER_SPREADDELTA_ANALYSIS_TUPLE,
                                            STC.POS_GEN_OVER_EDGEDELTA_ANALYSIS_TUPLE,
                                            STC.POS_GEN_OVER_COORDDELTA_ANALYSIS_TUPLE],
                                 line_3d_list=[STC.POS_GEN_OVER_SPREADDELTA_ANALYSIS_TUPLE,
                                               STC.POS_GEN_OVER_EDGEDELTA_ANALYSIS_TUPLE,
                                               STC.POS_GEN_OVER_COORDDELTA_ANALYSIS_TUPLE])
        self.__cardinal_analysis(pos_filter, prefix=p,
                                 line_table_list=[STC.POS_GEN_OVER_TIMEDELTA_ANALYSIS_TUPLE], force_filter=True)

    def clusterized_data_analysis(self):
        log.d("Test {} clusterized data analysis".format(DATASET_NAME))
        if not self.is_clustering_processed():
            log.e("Test {} clusterized data analysis: you have to process clustering earlier".format(DATASET_NAME))
            return self

        prefix = CLUSTER_IMG_FN_PREFIX
        prefix = "" if prefix is None else "{}_".format(prefix)

        dataset_for_clustering = self.data_clusterized
        self.__line_joint_analysis(self.__filter(dataset_for_clustering), line_list=[STC.CLUSTER_ANALYSIS_TUPLE],
                                   prefix=prefix + "all")
        self.__cardinal_analysis(dataset_for_clustering, line_list=[STC.CLUSTER_ANALYSIS_TUPLE],
                                 prefix=prefix + "all")

        # Clustering analysis for each partition
        partitions = self.partitions_clusterized
        for key in partitions:
            self.__line_joint_analysis(self.__filter(partitions[key]), prefix=prefix + key,
                                       line_list=[STC.CLUSTER_ANALYSIS_TUPLE],
                                       line_3d_list=[STC.CLUSTER_ANALYSIS_TUPLE])

    def maps(self):
        log.d("Test {} generate maps".format(DATASET_NAME))
        if not self.is_heuristic_processed():
            log.e("Test {} maps: you have to process heuristic earlier".format(DATASET_NAME))
            return self
        dataset_to_analyze = self.__pos_filter(self.st.pos)
        dataset_to_analyze = self.__filter(dataset_to_analyze)
        da_dataset = DataAnalysis(dataset_to_analyze, dataset_to_analyze.columns, dataset_name=DATASET_NAME,
                                  save_file=SAVE_FILE)
        da_dataset.show_line_map(on=STC.POS_GEN_OVER_RENTAL_MAP_TUPLE,
                                 hover_data=STC.POS_GEN_OVER_RENTAL_MAP_HOVER_DATA, groupby=self.groupby)
        da_dataset.show_scatter_map(on=STC.POS_GEN_OVER_COORDDELTA_MAP_TUPLE,
                                    hover_data=STC.POS_GEN_OVER_COORDDELTA_MAP_HOVER_DATA)

    def maps_3d(self):
        dataset_to_analyze = self.__pos_filter(self.st.pos)
        partitions = self.__partition(dataset_to_analyze, only_north=self.only_north)
        for key in partitions:
            partitions[key] = self.__filter(partitions[key])
            da_dataset = DataAnalysis(partitions[key], partitions[key].columns, dataset_name=DATASET_NAME,
                                      save_file=SAVE_FILE)
            filename = "{}_3d_map.html".format(key)
            da_dataset.show_3d_map(on=STC.POS_GEN_OVER_COORDDELTA_MAP_TUPLE, groupby=self.groupby, filename=filename)

    def stats(self):
        log.d("Test {} print stats".format(DATASET_NAME))
        self.st.print_stats()
        return self

    def partition_stats(self):
        partitions = self.__partition(self.st.pos, only_north=self.only_north)
        log.i("**************************** Scooter Trajectories Test - Partition Stats ******************************")
        for key in partitions:
            log.i("[PARTITION {} SHAPE]: {};".format(key, partitions[key].shape))
        log.i("*******************************************************************************************************")
        return self

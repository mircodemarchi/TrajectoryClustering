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
SAVE_FILE = True

RENTAL_IMG_FN_PREFIX = "rental"
POS_IMG_FN_PREFIX = "pos"
POS_OVER_RENTAL_IMG_FN_PREFIX = "all"
CLUSTER_IMG_FN_PREFIX = "clustering"
CLUSTER_TEST_IMG_FN_PREFIX = "clustering_test"
HEURISTIC_IMG_FN_PREFIX = "heuristic"

FORCE_RENTAL_NUM_TO_ANALYZE = 100

CLUSTERING_METHODS = ["k-means", "mean-shift", "gaussian-mixture", "full-agglomerative", "ward-agglomerative"]
CLUSTERING_EXAM_METHODS = ["k-means", "mean-shift", "ward-agglomerative"]
POS_NUM_FOR_AGGLOMERATIVE_CLUSTERING = 30000


class ScooterTrajectoriesTest:
    def __init__(self, log_lvl=None, chunk_size=None, max_chunk_num=None, rental_num_to_analyze=None,
                 timedelta=None, spreaddelta=None, edgedelta=None, group_on_timedelta=True,
                 n_clusters=None, with_pca=False, with_standardization=False, with_normalization=False,
                 only_north=False, exam=False):
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

        self.exam = exam

        # Others
        self.data_prepared = None
        self.partitions = None
        self.clustering_done = False
        self.all_clusters = None
        self.partitions_clusters = None
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

    def __pos_filter(self, pos):
        return pos[(pos[STC.POS_GEN_LONGITUDE_CN] > 8.0) & (pos[STC.POS_GEN_LONGITUDE_CN] < 15.0) &
                   (pos[STC.POS_GEN_LATITUDE_CN] > 42.0) & (pos[STC.POS_GEN_LATITUDE_CN] < 45.0)]

    def __rental_filter(self, rental):
        return rental[(rental[STC.RENTAL_START_LONGITUDE_CN] > 7.0) & (rental[STC.RENTAL_START_LONGITUDE_CN] < 15.0) &
                      (rental[STC.RENTAL_STOP_LONGITUDE_CN] > 7.0) & (rental[STC.RENTAL_STOP_LONGITUDE_CN] < 15.0) &
                      (rental[STC.RENTAL_START_LATITUDE_CN] > 42.0) & (rental[STC.RENTAL_START_LATITUDE_CN] < 46.0) &
                      (rental[STC.RENTAL_STOP_LATITUDE_CN] > 42.0) & (rental[STC.RENTAL_STOP_LATITUDE_CN] < 46.0)]

    def __filter(self, dataset, force=False):
        if force:
            rental_num = FORCE_RENTAL_NUM_TO_ANALYZE
        elif self.rental_num_to_analyze is not None:
            rental_num = self.rental_num_to_analyze
        else:
            return dataset
        rental_to_analyze = dataset[STC.MERGE_RENTAL_ID_CN].unique()[:rental_num]
        return dataset.loc[dataset[STC.MERGE_RENTAL_ID_CN].isin(rental_to_analyze)]

    def __prepare(self):
        log.d("Test {} prepare data for clustering".format(DATASET_NAME))
        start = time.time()
        merge_cols_pos_gen_map = dict(zip(STC.POS_GEN_COLS_MERGE_MAP.values(), STC.POS_GEN_COLS_MERGE_MAP.keys()))
        pos_sort_cols = STC.POS_GEN_SORT_COLS
        merge_sort_cols = [merge_cols_pos_gen_map[c] for c in pos_sort_cols]

        # Sort to align the frame rows
        ordered_merge = self.__pos_filter(self.st.merge)
        ordered_merge = ordered_merge.sort_values(by=merge_sort_cols, ignore_index=True)
        join = ordered_merge
        if self.is_heuristic_processed():
            ordered_pos = self.__pos_filter(self.st.pos)
            ordered_pos = ordered_pos.sort_values(by=pos_sort_cols, ignore_index=True)
            join = pd.concat([join, ordered_pos[STC.POS_GEN_HEURISTIC_COLS]], axis=1)

            # Cumsum the timedelta id
            group_on_timedelta = [STC.POS_GEN_RENTAL_ID_CN, STC.POS_GEN_TIMEDELTA_ID_CN]
            join[STC.POS_GEN_TIMEDELTA_ID_CN] = join.loc[:, group_on_timedelta].ne(
                join.loc[:, group_on_timedelta].shift()).any(axis=1).cumsum()

        # Convert time columns in float
        join_time_cols = STC.MERGE_TIME_COLS
        join[join_time_cols] = join[join_time_cols].applymap(lambda x: x.timestamp())

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))
        return join

    def __line_joint_analysis(self, dataset, joint_list=None, line_list=None, line_table_list=None, line_3d_list=None,
                              prefix=None):
        da = DataAnalysis(dataset, dataset.columns, dataset_name=DATASET_NAME, save_file=SAVE_FILE, prefix=prefix)
        if joint_list is not None:
            for t in joint_list:
                da.show_joint(on=t)

        if line_list is not None:
            for t in line_list:
                da.show_line(on=t, groupby=self.groupby)

        if line_table_list is not None:
            for t in line_table_list:
                da.show_line_table(on=t)

        if line_3d_list is not None:
            for t in line_list:
                da.show_3d_line(on=t, groupby=self.groupby)
        return self

    def __overview_analysis(self, dataset, cols, couples=None, prefix=None):
        da = DataAnalysis(dataset, cols, dataset_name=DATASET_NAME, save_file=SAVE_FILE, prefix=prefix)
        da.show_distributions()
        if couples is not None:
            da.show_2d_distributions(couples)
            for c in couples:
                da.show_joint(on=c)
        return self

    def __cardinal_analysis(self, dataset, joint_list=None, line_list=None, line_table_list=None, line_3d_list=None,
                            prefix=None, force_filter=False):
        partitions = self.__partition(dataset, only_north=self.only_north)
        for key in partitions:
            prefix_with_cardinal = "{}_{}".format(prefix, key) if prefix is not None else key
            partitions[key] = self.__filter(partitions[key], force_filter)
            self.__line_joint_analysis(partitions[key], joint_list=joint_list, line_list=line_list,
                                       line_table_list=line_table_list, line_3d_list=line_3d_list,
                                       prefix=prefix_with_cardinal)
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
        return self

    def load_from_generated(self):
        log.d("Test {} start load from already generated data".format(DATASET_NAME))
        if self.is_data_processed():
            log.w("Test {} load_from_generated: already processed".format(DATASET_NAME))
            return self

        self.st.load_generated()
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
        return self.clustering_done and self.all_clusters is not None and self.partitions_clusters is not None

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
        kmeans = Clustering(dataset_for_clustering, STC.CLUSTERING_COLS, dataset_name=DATASET_NAME)
        kmeans.test("k-means", range_clusters=range(1, 20), standardize=self.with_standardization,
                    normalize=self.with_normalization,  pca=self.with_pca, components=STC.CLUSTERING_COMPONENTS)
        kmeans.show_wcss(save_file=SAVE_FILE, prefix=prefix + "all")
        for key in partitions:
            kmeans = Clustering(partitions[key], STC.CLUSTERING_COLS, dataset_name=DATASET_NAME)
            kmeans.test("k-means", range_clusters=range(1, 30), standardize=self.with_standardization,
                        normalize=self.with_normalization, pca=self.with_pca, components=STC.CLUSTERING_COMPONENTS)
            kmeans.show_wcss(save_file=SAVE_FILE, prefix=prefix + key)

    def clustering(self):
        if self.n_clusters is None:
            self.test_clustering()
            return self

        log.d("Test {} start clustering".format(DATASET_NAME))
        if not self.is_heuristic_processed():
            log.e("Test {} clustering: you have to process heuristic earlier".format(DATASET_NAME))
            return self

        if self.exam:
            clustering_methods = CLUSTERING_EXAM_METHODS
        else:
            clustering_methods = CLUSTERING_METHODS

        components = STC.CLUSTERING_COMPONENTS  # STC.CLUSTERING_COMPONENTS or None or a number
        dataset_for_clustering = self.__prepare()
        self.data_prepared = dataset_for_clustering

        # Perform clustering in relation to each partition
        partitions = self.__partition(dataset_for_clustering, only_north=True if self.exam else self.only_north)
        self.partitions = partitions
        self.partitions_clusters = dict()
        for method in clustering_methods:
            self.partitions_clusters[method] = dict()
            for key in partitions:
                log.d("Test {} clustering of {} data with {}".format(DATASET_NAME, key, method))
                if method.endswith("agglomerative") and POS_NUM_FOR_AGGLOMERATIVE_CLUSTERING is not None:
                    c = Clustering(partitions[key].iloc[:POS_NUM_FOR_AGGLOMERATIVE_CLUSTERING],
                                   STC.CLUSTERING_COLS, dataset_name=DATASET_NAME)
                else:
                    c = Clustering(partitions[key], STC.CLUSTERING_COLS, dataset_name=DATASET_NAME)
                c.exec(method=method, n_clusters=self.n_clusters,
                       standardize=self.with_standardization, normalize=self.with_normalization,
                       pca=self.with_pca, components=components)
                self.partitions_clusters[method][key] = c

        self.all_clusters = dict()
        if self.exam:
            self.clustering_done = True
            return self
        # Perform clustering in relation to the entire data
        for method in clustering_methods:
            log.d("Test {} clustering of entire data with {}".format(DATASET_NAME, method))
            if method.endswith("agglomerative") and POS_NUM_FOR_AGGLOMERATIVE_CLUSTERING is not None:
                c = Clustering(dataset_for_clustering.iloc[:POS_NUM_FOR_AGGLOMERATIVE_CLUSTERING],
                               STC.CLUSTERING_COLS, dataset_name=DATASET_NAME)
            else:
                c = Clustering(dataset_for_clustering, STC.CLUSTERING_COLS, dataset_name=DATASET_NAME)
            c.exec(method=method, n_clusters=self.n_clusters,
                   standardize=self.with_standardization, normalize=self.with_normalization,
                   pca=self.with_pca, components=components)
            self.all_clusters[method] = c

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
        # Timedelta analysis
        self.__cardinal_analysis(pos_filter, prefix=p,
                                 line_table_list=[STC.POS_GEN_OVER_TIMEDELTA_TABLE_ANALYSIS_TUPLE], force_filter=True)
        group_on_timedelta = [STC.POS_GEN_RENTAL_ID_CN, STC.POS_GEN_TIMEDELTA_ID_CN]
        pos_unique_timedelta = pos_filter.copy()
        pos_unique_timedelta[STC.POS_GEN_TIMEDELTA_ID_CN] = pos_unique_timedelta.loc[:, group_on_timedelta].ne(
            pos_unique_timedelta.loc[:, group_on_timedelta].shift()).any(axis=1).cumsum()
        self.__cardinal_analysis(pos_unique_timedelta, prefix=p,
                                 line_list=[STC.POS_GEN_OVER_TIMEDELTA_ANALYSIS_TUPLE])

    def clusterized_data_analysis(self):
        log.d("Test {} analysis of clusterized data".format(DATASET_NAME))
        if not self.is_clustering_processed():
            log.e("Test {} clusterized data analysis: you have to process clustering earlier".format(DATASET_NAME))
            return self

        partitions = self.partitions
        for method in self.partitions_clusters:
            prefix = "{}_{}_".format(CLUSTER_IMG_FN_PREFIX, method)
            # Clustering analysis for each partition
            for key in partitions:
                log.d("Test {} analysis clusterized of {} data with {}".format(DATASET_NAME, key, method))
                self.partitions_clusters[method][key].stats()
                if method.endswith("agglomerative") and POS_NUM_FOR_AGGLOMERATIVE_CLUSTERING is not None:
                    p = partitions[key].iloc[:POS_NUM_FOR_AGGLOMERATIVE_CLUSTERING].copy()
                    self.partitions_clusters[method][key].show_dendrogram(prefix=prefix + key, save_file=SAVE_FILE)
                else:
                    p = partitions[key].copy()
                p[STC.CLUSTER_ID_CN] = self.partitions_clusters[method][key].labels
                self.__line_joint_analysis(self.__filter(p), prefix=prefix + key,
                                           line_list=[STC.CLUSTER_ANALYSIS_TUPLE],
                                           line_3d_list=[STC.CLUSTER_ANALYSIS_TUPLE])
        dataset_for_clustering = self.data_prepared
        for method in self.all_clusters:
            log.d("Test {} analysis clusterized of entire data with {}".format(DATASET_NAME, method))
            prefix = "{}_{}_".format(CLUSTER_IMG_FN_PREFIX, method)
            self.all_clusters[method].stats()
            # Clustering analysis for the entire dataset
            if method.endswith("agglomerative") and POS_NUM_FOR_AGGLOMERATIVE_CLUSTERING is not None:
                d = dataset_for_clustering.iloc[:POS_NUM_FOR_AGGLOMERATIVE_CLUSTERING].copy()
                self.all_clusters[method].show_dendrogram(prefix=prefix + "all", save_file=SAVE_FILE)
            else:
                d = dataset_for_clustering.copy()
            d[STC.CLUSTER_ID_CN] = self.all_clusters[method].labels
            self.__line_joint_analysis(self.__filter(d), line_list=[STC.CLUSTER_ANALYSIS_TUPLE], prefix=prefix + "all")
            self.__cardinal_analysis(d, line_list=[STC.CLUSTER_ANALYSIS_TUPLE], prefix=prefix + "all")

    def maps(self):
        log.d("Test {} generate maps".format(DATASET_NAME))
        if not self.is_heuristic_processed():
            log.e("Test {} maps: you have to process heuristic earlier".format(DATASET_NAME))
            return self
        dataset_to_analyze = self.__pos_filter(self.st.pos)
        dataset_to_analyze = self.__filter(dataset_to_analyze).copy()
        dataset_to_analyze[STC.POS_GEN_LATITUDE_CN] += 1.2
        dataset_to_analyze[STC.POS_GEN_LONGITUDE_CN] += -1.2

        group_on_timedelta = [STC.POS_GEN_RENTAL_ID_CN, STC.POS_GEN_TIMEDELTA_ID_CN]
        dataset_to_analyze[STC.POS_GEN_TIMEDELTA_ID_CN] = dataset_to_analyze.loc[:, group_on_timedelta].ne(
            dataset_to_analyze.loc[:, group_on_timedelta].shift()).any(axis=1).cumsum()

        da_dataset = DataAnalysis(dataset_to_analyze, dataset_to_analyze.columns, dataset_name=DATASET_NAME,
                                  save_file=SAVE_FILE)
        da_dataset.show_scatter_map(on=STC.POS_GEN_OVER_RENTAL_MAP_TUPLE,
                                    hover_data=STC.POS_GEN_OVER_RENTAL_MAP_HOVER_DATA,
                                    filename="rental_scatter_map.html")
        if not self.exam:
            da_dataset.show_scatter_map(on=STC.POS_GEN_OVER_COORDDELTA_MAP_TUPLE,
                                        hover_data=STC.POS_GEN_OVER_COORDDELTA_MAP_HOVER_DATA,
                                        filename="coord_scatter_map.html")
            da_dataset.show_scatter_map(on=STC.POS_GEN_OVER_TIMEDELTA_MAP_TUPLE,
                                        hover_data=STC.POS_GEN_OVER_TIMEDELTA_MAP_HOVER_DATA,
                                        filename="timedelta_scatter_map.html")

    def cluster_maps(self):
        key = "N-E"
        method = "k-means"
        log.d("Test {} generate maps for clustering".format(DATASET_NAME))
        if not self.is_clustering_processed():
            log.e("Test {} maps for clustering: you have to process clustering earlier".format(DATASET_NAME))
            return self
        partitions = self.partitions
        p = partitions[key].copy()
        p[STC.CLUSTER_ID_CN] = self.partitions_clusters[method][key].labels
        p = self.__filter(p)
        p[STC.POS_GEN_LATITUDE_CN] += 1.2
        p[STC.POS_GEN_LONGITUDE_CN] += -1.2
        da_dataset = DataAnalysis(p, p.columns, dataset_name=DATASET_NAME, save_file=SAVE_FILE)
        da_dataset.show_scatter_map(on=STC.POS_GEN_OVER_CLUSTER_MAP_TUPLE,
                                    hover_data=STC.POS_GEN_OVER_CLUSTER_MAP_HOVER_DATA,
                                    filename="cluster_scatter_map.html")

    def cluster_maps_3d(self):
        key = "N-E"
        method = "k-means"
        log.d("Test {} generate maps 3D for clustering".format(DATASET_NAME))
        if not self.is_clustering_processed():
            log.e("Test {} maps 3D for clustering: you have to process clustering earlier".format(DATASET_NAME))
            return self
        partitions = self.partitions
        p = partitions[key]
        p[STC.CLUSTER_ID_CN] = self.partitions_clusters[method][key].labels
        p = self.__filter(p)
        da_dataset = DataAnalysis(p, p.columns, dataset_name=DATASET_NAME, save_file=SAVE_FILE)
        filename = "{}_{}_cluster_3d_map.html".format(key, method)
        da_dataset.show_3d_map(on=STC.POS_GEN_OVER_CLUSTER_MAP_TUPLE,
                               hover_data=STC.POS_GEN_OVER_CLUSTER_MAP_HOVER_DATA,
                               groupby=self.groupby, filename=filename)

    def stats(self):
        log.d("Test {} print stats".format(DATASET_NAME))
        self.st.print_stats()
        return self

    def partition_stats(self):
        partitions = self.__partition(self.st.pos, only_north=self.only_north)
        log.i("**************************** Scooter Trajectories Test - Partition Stats ******************************")
        for key in partitions:
            log.i("[PARTITION {} SHAPE]: {};".format(key, partitions[key].shape))
            log.i("[PARTITION {} DESCRIPTION]: \n{}\n{};".format(key,
                  partitions[key][partitions[key].columns[
                                :int(len(partitions[key].columns) / 2)]].describe(datetime_is_numeric=True),
                  partitions[key][partitions[key].columns[
                                int(len(partitions[key].columns) / 2):]].describe(datetime_is_numeric=True)))
        log.i("[FILTER POS SHAPE]: {}".format(len(self.__filter(self.__pos_filter(self.st.pos)).index)))
        log.i("[FILTER RENTAL SHAPE]: {}".format(self.rental_num_to_analyze if self.rental_num_to_analyze is not None
                                                 else len(self.__rental_filter(self.st.rental).index)))
        log.i("*******************************************************************************************************")
        return self

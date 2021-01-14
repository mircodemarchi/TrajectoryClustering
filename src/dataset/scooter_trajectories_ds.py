import pandas as pd
import os
import time
import json

from util.log import Log
from util.util import get_elapsed, unzip
from util.constant import DATA_FOLDER

from .constant import ScooterTrajectoriesC as C


log = Log(__name__, enable_console=True, enable_file=False)


class ScooterTrajectoriesDS:
    def __init__(self, zip_filepath=None, log_lvl=None):
        if zip_filepath:
            self.zip_filepath = zip_filepath
        else:
            self.zip_filepath = os.path.join(DATA_FOLDER, C.ZIP_DEFAULT_FN)

        if log_lvl is not None:
            if not isinstance(log_lvl, int):
                raise ValueError("Invalid log level: {}".format(log_lvl))
            log.set_level(log_lvl)

        self.unzip_folder = os.path.splitext(self.zip_filepath)[0]
        self.merge = pd.DataFrame(columns=C.MERGE_COLS)
        self.dataset = pd.DataFrame(columns=C.DATASET_COLS)
        self.rental = pd.DataFrame(columns=C.RENTAL_COLS)
        self.pos = pd.DataFrame(columns=C.POS_GEN_COLS)

    def __unzip(self):
        if not os.path.exists(self.zip_filepath):
            log.f("Unzip path not exist")
            return

        unzip(self.zip_filepath)

    def __is_pos_timestamp_in_rental(self, pos, rental):
        pos_server_time_cn = C.POS_TIME_COLS[0]
        pos_device_time_cn = C.POS_TIME_COLS[1]
        rental_start_time_cn = C.RENTAL_TIME_COLS[0]
        rental_end_time_cn = C.RENTAL_TIME_COLS[1]

        return (pos[pos_server_time_cn] >= rental[rental_start_time_cn]) & \
               (pos[pos_device_time_cn] >= rental[rental_start_time_cn]) & \
               (pos[pos_server_time_cn] <= rental[rental_end_time_cn]) & \
               (pos[pos_device_time_cn] <= rental[rental_end_time_cn])

    def __parse_pos_datetime(self, df):
        for date_cn in C.POS_TIME_COLS:
            df[date_cn] = pd.to_datetime(df[date_cn])
        return df

    def __parse_rental_datetime(self, df):
        for date_cn in C.RENTAL_TIME_COLS:
            df[date_cn] = pd.to_datetime(df[date_cn])
        return df

    def __load_device_data(self):
        if not os.path.exists(self.unzip_folder):
            log.e("Scooter Trajectories folder not extracted: impossible to load device data")
            return pd.DataFrame(), get_elapsed(0, 0)

        start = time.time()
        # Parse device data
        device_file = os.path.join(self.unzip_folder, C.CSV_DEVICE_FN)
        device_data = pd.read_csv(device_file, memory_map=True, names=C.DEVICE_COLS, header=0)
        end = time.time()

        return device_data, get_elapsed(start, end)

    def __load_pos_data(self, chunknum=None, chunksize=50000, max_chunknum=None):
        if not os.path.exists(self.unzip_folder):
            log.e("Scooter Trajectories folder not extracted: impossible to load position data")
            return pd.DataFrame(), get_elapsed(0, 0)

        start = time.time()
        pos_data = pd.DataFrame()

        # Parse device data
        curr_chunk = 0
        for fn in C.CSV_POS_FN:
            header = 0  # if fn == self.CSV_POS_FN[0] else None
            pos_file = os.path.join(self.unzip_folder, fn)
            reader = pd.read_csv(pos_file, names=C.POS_COLS, header=header, chunksize=chunksize,
                                 iterator=True, memory_map=True)
            for pos_chunk_df in reader:
                if chunknum is not None:
                    if chunknum == curr_chunk:
                        end = time.time()
                        reader.close()
                        return self.__parse_pos_datetime(pos_chunk_df), get_elapsed(start, end)
                else:
                    # Load all chunk
                    pos_data = pd.concat([pos_data, pos_chunk_df], axis=0)

                curr_chunk += 1

                if max_chunknum == curr_chunk:
                    end = time.time()
                    return self.__parse_pos_datetime(pos_data), get_elapsed(start, end)

            reader.close()

        end = time.time()
        return self.__parse_pos_datetime(pos_data), get_elapsed(start, end)

    def __load_rental_data(self):
        if not os.path.exists(self.unzip_folder):
            log.e("Scooter Trajectories folder not extracted: impossible to load rental data")
            return pd.DataFrame(), get_elapsed(0, 0)

        start = time.time()
        # Parse device data
        rental_file = os.path.join(self.unzip_folder, C.CSV_RENTAL_FN)
        rental_data = pd.read_csv(rental_file, memory_map=True, names=C.RENTAL_COLS, header=0)
        rental_data = self.__parse_rental_datetime(rental_data)

        end = time.time()
        return rental_data, get_elapsed(start, end)

    def __load_user_data(self):
        if not os.path.exists(self.unzip_folder):
            log.e("Scooter Trajectories folder not extracted: impossible to load user data")
            return pd.DataFrame(), get_elapsed(0, 0)

        start = time.time()
        # Parse device data
        user_file = os.path.join(self.unzip_folder, C.CSV_USR_FN)
        user_data = pd.read_csv(user_file, memory_map=True, names=C.USR_COLS, header=0)

        end = time.time()
        return user_data, get_elapsed(start, end)

    def __generate(self, device_df, rental_df, user_df, chunknum=None, chunksize=50000, max_chunknum=None):
        if not os.path.exists(self.unzip_folder):
            log.e("Scooter Trajectories folder not extracted: impossible to load pos and rental map data")
            return pd.DataFrame(), get_elapsed(0, 0)

        start = time.time()
        gen_data = pd.DataFrame(columns=C.MERGE_COLS)

        # Parse device data
        curr_chunk = 0
        for fn in C.CSV_POS_FN:
            header = 0  # if fn == self.CSV_POS_FN[0] else None
            pos_file = os.path.join(self.unzip_folder, fn)
            reader = pd.read_csv(pos_file, names=C.POS_COLS, header=header, chunksize=chunksize, iterator=True,
                                 memory_map=True)

            for pos_chunk_df in reader:
                # Filter pos and rental data
                pos_chunk_df, rental_df, _ = self.__filter_rental_pos(pos_chunk_df, rental_df,
                                                                      user_df, device_df)

                # Combine positions to related rental
                # self.dataset = self.__map_pos_into_rental(rental_filtered_df, pos_filtered_df)
                # self.dataset = self.__map_rental_into_pos(rental_filtered_df, pos_filtered_df)
                # pos_rental_map_df, merge_elapsed_time = self.__merge_rental_into_pos(pos_filtered_df,
                #                                                                      rental_filtered_df)
                if chunknum is not None:
                    if chunknum == curr_chunk:
                        pos_chunk_df = self.__parse_pos_datetime(pos_chunk_df)
                        pos_rental_map_df, merge_elapsed_time = self.__merge(pos_chunk_df, rental_df,
                                                                             user_df, device_df)
                        log.d("__chunk {} merge elapsed time: {}".format(curr_chunk, merge_elapsed_time))
                        reader.close()
                        end = time.time()
                        return pos_rental_map_df, get_elapsed(start, end)
                else:
                    # Load all chunk
                    pos_chunk_df = self.__parse_pos_datetime(pos_chunk_df)
                    pos_rental_map_df, merge_elapsed_time = self.__merge(pos_chunk_df, rental_df,
                                                                         user_df, device_df)
                    log.d("__chunk {} merge elapsed time: {}".format(curr_chunk, merge_elapsed_time))
                    gen_data = pd.concat([gen_data, pos_rental_map_df], axis=0)

                    if max_chunknum is not None and max_chunknum == curr_chunk:
                        end = time.time()
                        return gen_data, get_elapsed(start, end)

                curr_chunk += 1

            reader.close()

        end = time.time()
        return gen_data, get_elapsed(start, end)

    def __build(self, gen_df, rental_df):
        start = time.time()
        # Calculate valid positions and rental according to the timestamp of generated data
        pos_df = gen_df[list(C.POS_GEN_COLS_MERGE_MAP.keys())]
        pos_df = pos_df.rename(columns=C.POS_GEN_COLS_MERGE_MAP)
        pos_df, rental_df = self.__find_data_map(gen_df, pos_df, rental_df)

        # Sort generated data, pos data and rental data
        pos_df = pd.DataFrame(pos_df, columns=C.POS_GEN_COLS)
        merge_df, pos_df, rental_df, _ = self.__sort(gen_df, pos_df, rental_df)

        # Build dataset
        dataset_group_cols = list(C.MERGE_COLS_RENTAL_MAP.values()) + list(C.MERGE_COLS_DEVICE_MAP.values()) + list(
            C.MERGE_COLS_USR_MAP.values())
        dataset_group_cols = list(dict.fromkeys(dataset_group_cols))
        groups_by_rental = merge_df.groupby(dataset_group_cols)
        dataset_df = groups_by_rental[C.MERGE_POS_ID_CN].apply(list).reset_index(name=C.DATASET_RENTAL_POSITIONS_CN)
        dataset_df = dataset_df[C.DATASET_COLS]

        end = time.time()
        return dataset_df, merge_df, pos_df, rental_df, get_elapsed(start, end)

    def __map_pos_into_rental(self, rental_df: pd.DataFrame, pos_df: pd.DataFrame):
        def ___find_pos_of_rental(p_df, rental, rental_idx):
            start = time.time()
            pos_of_rental = p_df.loc[
                (p_df[C.POS_DEVICE_ID_CN] == rental[C.RENTAL_DEVICE_ID_CN])
                & self.__is_pos_timestamp_in_rental(p_df, rental)]
            end = time.time()
            print("__rental {}; positions found: {}; time: {};"
                  .format(rental_idx, pos_of_rental.index.to_numpy(), get_elapsed(start, end)))
            return pos_of_rental

        rental_df[C.DATASET_RENTAL_POSITIONS_CN] = [
            ___find_pos_of_rental(pos_df, rental, rental_idx)
            for rental_idx, rental in rental_df.iterrows()]

        # Take only the rental with positions not empty
        not_empty_positions = [pos.size != 0 for pos in rental_df[C.DATASET_RENTAL_POSITIONS_CN]]
        rental_df = rental_df[not_empty_positions]
        return rental_df

    def __map_rental_into_pos(self, rental_df: pd.DataFrame, pos_df: pd.DataFrame):
        def ___find_rental_of_pos(r_df, pos, pos_idx):
            rental_of_pos = r_df.loc[
                (rental_df[C.RENTAL_DEVICE_ID_CN] == pos[C.POS_DEVICE_ID_CN])
                & self.__is_pos_timestamp_in_rental(pos, r_df)].to_numpy()
            if pos_idx % 1000 == 0:
                log.d("__position {:10} processed".format(pos_idx))
            return rental_of_pos

        pos_df[C.POS_RENTAL_CN] = [___find_rental_of_pos(rental_df, pos, pos_idx)
                                      for pos_idx, pos in pos_df.iterrows()]

        # Remove positions without rentals
        pos_df = pos_df[[rentals.size != 0 for rentals in pos_df[C.POS_RENTAL_CN]]]

        return pos_df

    def __merge(self, pos_df: pd.DataFrame, rental_df: pd.DataFrame, user_df: pd.DataFrame, device_df: pd.DataFrame):
        start = time.time()

        # Merge rental data with pos data
        rental_df = rental_df.rename(columns=C.MERGE_COLS_RENTAL_MAP)
        pos_df = pos_df.rename(columns=C.MERGE_COLS_POS_MAP)
        merged_df = rental_df.merge(pos_df, on=C.POS_DEVICE_ID_CN)

        # Filter pos data in rental timestamp range
        merged_df = merged_df.loc[
            self.__is_pos_timestamp_in_rental(merged_df[C.POS_TIME_COLS], merged_df[C.RENTAL_TIME_COLS])]

        # Merge rental data with usr data
        user_df = user_df.rename(columns=C.MERGE_COLS_USR_MAP)
        merged_df = merged_df.merge(user_df, on=C.RENTAL_USR_ID_CN)

        # Merge rental data with device data
        device_df = device_df.rename(columns=C.MERGE_COLS_DEVICE_MAP)
        merged_df = merged_df.merge(device_df, on=C.RENTAL_DEVICE_ID_CN)

        end = time.time()
        return merged_df, get_elapsed(start, end)

    def __filter_rental_pos(self, pos_df, rental_df, user_df, device_df):
        start = time.time()
        # Filter rental according to user id and device id
        rental_filtered_df = rental_df.loc[rental_df[C.RENTAL_DEVICE_ID_CN].isin(device_df[C.DEVICE_ID_CN]) &
                                           rental_df[C.RENTAL_USR_ID_CN].isin(user_df[C.USR_ID_CN]) &
                                           rental_df[C.RENTAL_DEVICE_ID_CN].isin(pos_df[C.POS_DEVICE_ID_CN])]

        # Filter positions according to rentals
        pos_filtered_df = pos_df.loc[pos_df[C.POS_DEVICE_ID_CN].isin(rental_filtered_df[C.RENTAL_DEVICE_ID_CN])]

        end = time.time()
        return pos_filtered_df, rental_filtered_df, get_elapsed(start, end)

    def __find_data_map(self, rental_pos_map_df, pos_df, rental_df):
        rental_valid_df = rental_df.loc[rental_df[C.RENTAL_ID_CN].isin(rental_pos_map_df[C.MERGE_RENTAL_ID_CN])]
        pos_valid_df = pos_df.loc[pos_df[C.POS_GEN_ID_CN].isin(rental_pos_map_df[C.MERGE_POS_ID_CN])]
        return pos_valid_df, rental_valid_df

    def __find_timestamp_cluster(self, group_df, time_delta):
        pos_time_cols = [C.POS_GEN_SERVER_TIME_CN, C.POS_GEN_DEVICE_TIME_CN]
        time_columns_group = group_df[pos_time_cols].reset_index(drop=True)

        # Find the positions that have a distance between the previous position at least of time_distance
        prev_time_columns_group = time_columns_group.iloc[time_columns_group.index - 1].reset_index(drop=True)
        time_gaps = time_columns_group.subtract(prev_time_columns_group) >= time_delta

        # Assign a different id for each time sequence
        time_gaps = time_gaps[pos_time_cols[0]] & time_gaps[pos_time_cols[1]]
        return time_gaps.cumsum()

    def __load_support_data(self):
        start = time.time()
        device_df, _ = self.__load_device_data()
        rental_df, _ = self.__load_rental_data()
        user_df, _ = self.__load_user_data()
        end = time.time()
        return device_df, rental_df, user_df, get_elapsed(start, end)

    def __sort(self, rental_pos_map_df, pos_df, rental_df):
        start = time.time()
        # List column names for sorting
        merge_sort_cols = C.MERGE_SORT_COLS
        pos_sort_cols = C.POS_GEN_SORT_COLS
        rental_sort_cols = C.RENTAL_SORT_COLS

        # Sort generated merge
        rental_pos_map_df = rental_pos_map_df.sort_values(by=merge_sort_cols, ignore_index=True)
        pos_df = pos_df.sort_values(by=pos_sort_cols, ignore_index=True)
        rental_df = rental_df.sort_values(by=rental_sort_cols, ignore_index=True)

        # Columns sort
        rental_pos_map_df = rental_pos_map_df[C.MERGE_COLS]
        pos_df = pos_df[C.POS_GEN_COLS]
        rental_df = rental_df[C.RENTAL_COLS]

        end = time.time()
        return rental_pos_map_df, pos_df, rental_df, get_elapsed(start, end)

    def generate(self, chunknum=0, chunksize=50000):
        log.d("Scooter Trajectories start unzip")
        self.__unzip()

        log.d("Scooter Trajectories load device, rental, user data")
        device_df, rental_df, user_df, load_time = self.__load_support_data()
        log.d("elapsed time: {}".format(load_time))

        # Load dataset and positions according to chunk info
        log.d("Scooter Trajectories load pos and rental timestamp map data")
        gen_df, load_time = self.__generate(device_df, rental_df, user_df, chunknum=chunknum, chunksize=chunksize)
        log.d("elapsed time: {}".format(load_time))

        log.d("Scooter Trajectories build final data")
        self.dataset, self.merge, self.pos, self.rental, load_time = self.__build(gen_df, rental_df)
        log.d("elapsed time: {}".format(load_time))

        return self

    def generate_all(self, chunksize=50000, max_chunknum=None):
        log.d("Scooter Trajectories start unzip")
        self.__unzip()

        log.d("Scooter Trajectories load device, rental, user data")
        device_df, rental_df, user_df, load_time = self.__load_support_data()
        log.d("elapsed time: {}".format(load_time))

        log.d("Scooter Trajectories load pos and rental timestamp map data")
        gen_df, load_time = self.__generate(device_df, rental_df, user_df, chunknum=None,
                                            chunksize=chunksize, max_chunknum=max_chunknum)
        log.d("elapsed time: {}".format(load_time))

        log.d("Scooter Trajectories build final data")
        self.dataset, self.merge, self.pos, self.rental, load_time = self.__build(gen_df, rental_df)
        log.d("elapsed time: {}".format(load_time))

        return self

    def load_generated(self):
        if not os.path.exists(os.path.join(DATA_FOLDER, C.GENERATED_DN)):
            log.e("Generated folder not exist")
            return self

        log.d("Scooter Trajectories load generated")
        start = time.time()

        dataset_gen_fp = os.path.join(DATA_FOLDER, C.GENERATED_DN, C.CSV_DATASET_GENERATED_FN)
        if os.path.exists(dataset_gen_fp):
            self.dataset = pd.read_csv(dataset_gen_fp, memory_map=True, parse_dates=C.DATASET_TIME_COLS,
                                       converters={C.DATASET_RENTAL_POSITIONS_CN: json.loads})
        else:
            log.w("{} path not exist".format(dataset_gen_fp))

        merge_gen_fp = os.path.join(DATA_FOLDER, C.GENERATED_DN, C.CSV_MERGE_GENERATED_FN)
        if os.path.exists(merge_gen_fp):
            self.merge = pd.read_csv(merge_gen_fp, parse_dates=C.MERGE_TIME_COLS,
                                     infer_datetime_format=True, memory_map=True)
        else:
            log.w("{} path not exist".format(merge_gen_fp))

        pos_gen_fp = os.path.join(DATA_FOLDER, C.GENERATED_DN, C.CSV_POS_GENERATED_FN)
        if os.path.exists(pos_gen_fp):
            self.pos = pd.read_csv(pos_gen_fp, parse_dates=C.POS_TIME_COLS,
                                   infer_datetime_format=True, memory_map=True)
        else:
            log.w("{} path not exist".format(pos_gen_fp))

        rental_gen_fp = os.path.join(DATA_FOLDER, C.GENERATED_DN, C.CSV_RENTAL_GENERATED_FN)
        if os.path.exists(rental_gen_fp):
            self.rental = pd.read_csv(rental_gen_fp, parse_dates=C.RENTAL_TIME_COLS,
                                      infer_datetime_format=True, memory_map=True)
        else:
            log.w("{} path not exist".format(rental_gen_fp))

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))

        return self

    def to_csv(self):
        log.d("Scooter Trajectories to csv")
        start = time.time()

        if not os.path.exists(os.path.join(DATA_FOLDER, C.GENERATED_DN)):
            os.makedirs(os.path.join(DATA_FOLDER, C.GENERATED_DN))

        # Save data in csv files
        pos_gen_fp = os.path.join(DATA_FOLDER, C.GENERATED_DN, C.CSV_POS_GENERATED_FN)
        if not self.pos.empty:
            self.pos.to_csv(pos_gen_fp, index=False)

        rental_gen_fp = os.path.join(DATA_FOLDER, C.GENERATED_DN, C.CSV_RENTAL_GENERATED_FN)
        if not self.rental.empty:
            self.rental.to_csv(rental_gen_fp, index=False)

        merge_gen_fp = os.path.join(DATA_FOLDER, C.GENERATED_DN, C.CSV_MERGE_GENERATED_FN)
        if not self.merge.empty:
            self.merge.to_csv(merge_gen_fp, index=False)

        dataset_gen_fp = os.path.join(DATA_FOLDER, C.GENERATED_DN, C.CSV_DATASET_GENERATED_FN)
        if not self.dataset.empty:
            self.dataset.to_csv(dataset_gen_fp, index=False)

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))

        return self

    def timestamp_clustering(self, time_distance):
        log.d("Scooter Trajectories timestamp clustering")
        start = time.time()
        time_delta = pd.Timedelta(time_distance)

        groups_by_rental = self.pos.groupby(by=[C.POS_GEN_RENTAL_ID_CN])

        timestamp_cluster = []
        for _, group in groups_by_rental:
            timestamp_cluster.extend(self.__find_timestamp_cluster(group, time_delta))

        self.dataset[C.POS_GEN_CLUSTER_ID_CN] = timestamp_cluster

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))
        return self

    def print_stats(self):
        if self.dataset.empty or self.rental.empty or self.pos.empty:
            log.e("Empty dataframe: print_stats() error")
            return self

        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

        log.i("************************************ Scooter Trajectories - Stats *************************************")
        log.i("[DATA MAP SHAPE]: {};".format(self.dataset.shape))
        log.i("[DATA MAP COLUMN NAMES]: {};".format(list(self.dataset.columns)))
        log.i("[DATA MAP FEATURES TYPES]:\n{};".format(self.dataset.dtypes))
        log.i("[DATA MAP NULL OCCURRENCES]:\n{};".format(self.dataset.isnull().sum()))
        log.i("[DATA MAP DESCRIPTION]:\n{}\n{};".format(
            self.dataset[self.dataset.columns[:int(len(self.dataset.columns)/2)]].describe(datetime_is_numeric=True),
            self.dataset[self.dataset.columns[int(len(self.dataset.columns)/2):]].describe(datetime_is_numeric=True)))
        log.i("[RENTAL SHAPE]: {};".format(self.rental.shape))
        log.i("[RENTAL COLUMN NAMES]: {};".format(list(self.rental.columns)))
        log.i("[RENTAL FEATURES TYPES]:\n{};".format(self.rental.dtypes))
        log.i("[RENTAL NULL OCCURRENCES]:\n{};".format(self.rental.isnull().sum()))
        log.i("[RENTAL DESCRIPTION]:\n{};".format(self.rental.describe(datetime_is_numeric=True)))
        log.i("[POS SHAPE]: {};".format(self.pos.shape))
        log.i("[POS COLUMN NAMES]: {};".format(list(self.pos.columns)))
        log.i("[POS FEATURES TYPES]:\n{};".format(self.pos.dtypes))
        log.i("[POS NULL OCCURRENCES]:\n{};".format(self.pos.isnull().sum()))
        log.i("[POS DESCRIPTION]:\n{};".format(self.pos.describe(datetime_is_numeric=True)))
        log.i("*******************************************************************************************************")
        return self

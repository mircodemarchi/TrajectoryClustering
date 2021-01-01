import pandas as pd
import numpy as np
import zipfile
import os
import time
import sys
from tqdm import tqdm
from .log import Log
from .util import get_elapsed

DATA_FOLDER = "../data"

log = Log(__name__, enable_console=True, enable_file=False)


def get_folder_size(path="."):
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # Skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


def unzip(path):
    with zipfile.ZipFile(path, "r") as zf:

        # If unzip folder already exist, check same size
        target_dir = os.path.splitext(path)[0]
        if os.path.exists(target_dir):
            uncompress_size = sum((file.file_size for file in zf.infolist()))
            if get_folder_size(target_dir) == uncompress_size:
                gen = tqdm(desc=os.path.basename(path), total=len(zf.namelist()), file=sys.stdout)
                gen.update(len(zf.namelist()))
                gen.close()
                return

        # Extract
        gen = tqdm(desc=os.path.basename(path), iterable=zf.namelist(), total=len(zf.namelist()), file=sys.stdout)
        for file in gen:
            zf.extract(member=file, path=DATA_FOLDER)
        gen.close()

    # Another implementation of progress bar
    # import sys
    # path = os.path.join(DATA_FOLDER, name)
    # with zipfile.ZipFile(path, "r") as zf:
    #     uncompress_size = sum((file.file_size for file in zf.infolist()))
    #     extracted_size = 0
    #     for file in zf.infolist():
    #         extracted_size += file.file_size
    #         sys.stdout.write("\r[ %.3f %% ] : %s" % (extracted_size * 100 / uncompress_size, file.filename))
    #         zf.extract(member=file, path=DATA_FOLDER)
    #
    # sys.stdout.write("\rDownload completed: %s\n" % name)


class MotionSenseDS:
    TRIALS_NUM = 15
    TRIALS_LONG = range(1, 10)  # [1, 9]
    TRIALS_SHORT = range(11, 17)  # [11, 16]

    ZIP_DEFAULT_FN = "motion-sense.zip"
    SUBDIRNAME = "A_DeviceMotion_data"
    CSV_SUBJECT_INFO_FN = "data_subjects_info.csv"
    TRIAL_LONG_CN = "is long"
    SUBJECT_CN = "subject"
    TRIAL_TYPE_CN = "trial type"

    def __init__(self, unzip_path=None):
        if unzip_path:
            self.unzip_path = unzip_path
        else:
            self.unzip_path = os.path.join(DATA_FOLDER, self.ZIP_DEFAULT_FN)

        self.unzip_folder = os.path.splitext(self.unzip_path)[0]
        self.target = None
        self.dataset = pd.DataFrame()
        self.subject_info = pd.DataFrame()

    def __is_trial_long(self, trial_number):
        if trial_number in self.TRIALS_LONG:
            return True

        if trial_number in self.TRIALS_SHORT:
            return False

        return None

    def __unzip(self):
        if not os.path.exists(self.unzip_path):
            log.f("Unzip path not exist")
            return

        unzip(self.unzip_path)

    def __load_subject_info(self):
        if not os.path.exists(self.unzip_folder):
            log.e("Motion sense folder not extracted: impossible to load subject info")
            return

        # Parse subject info
        subject_info_file = os.path.join(self.unzip_folder, self.CSV_SUBJECT_INFO_FN)
        subject_info_data = pd.read_csv(subject_info_file)

        return subject_info_data

    def __load_sensor_data(self, mask):
        data_subfolder = os.path.join(self.unzip_folder, self.SUBDIRNAME)
        df = pd.DataFrame()
        i = 0
        for dirpath, dirnames, filenames in os.walk(data_subfolder):
            if data_subfolder != dirpath:
                # Extract trial type from the current dirname. The dirname basename pattern is
                # "<trial-type>_<trial-num>", then "<trial-type>_<trial-num>".split("_")[0]==<trial-type>.
                current_trial_type = os.path.basename(dirpath).split("_")[0]

                # Extract trial number from the current dirname. The dirname basename pattern is
                # "<trial-type>_<trial-num>", then "<trial-type>_<trial-num>".split("_")[0]==<trial-num>.
                current_trial_num = int(os.path.basename(dirpath).split("_")[1])

                # Extract data using the subject selection mask for the current trial.
                # Every CSV filename pattern is sub_<subject-num>.csv, the operation performed to get the <subject-num>
                # are os.path.splitext(fn)[0]=="sub_<subject-num>" and "sub_<subject-num>".split("_")[1]==<subject-num>.
                selected_subject = mask[i]
                selected_trial_file = [fn for fn in filenames
                                       if int(os.path.splitext(fn)[0].split("_")[1]) == selected_subject]
                if selected_trial_file:
                    data_trial = pd.read_csv(os.path.join(dirpath, selected_trial_file[0]))
                    del data_trial[data_trial.columns[0]]

                    data_trial[self.TRIAL_LONG_CN] = np.full(len(data_trial.index),
                                                             self.__is_trial_long(current_trial_num))
                    data_trial[self.SUBJECT_CN] = np.full(len(data_trial.index), selected_subject)
                    data_trial[self.TRIAL_TYPE_CN] = np.full(len(data_trial.index), current_trial_type)

                    df = df.append(data_trial)

        return df

    def load(self, mask=None):
        if mask is None or mask.shape[0] != self.TRIALS_NUM:
            mask = self.TRIALS_NUM * [0]

        # Unzip
        log.d("Motion sense start unzip")
        self.__unzip()

        # Parse subject info
        log.d("Motion sense start parse")
        subject_info = self.__load_subject_info()

        # Parse sensor data
        target = self.TRIAL_TYPE_CN
        sensor_data = self.__load_sensor_data(mask)

        # Merge sensor data with subject info
        if not sensor_data.empty:
            dataset = sensor_data.merge(subject_info,
                                        left_on=self.SUBJECT_CN,
                                        right_on=subject_info.columns[0])
            # del dataset[self.DATA_SUBJECT_COLUMN_NAME]
            del dataset[subject_info.columns[0]]
        else:
            dataset = sensor_data

        log.i("Motion sense dataset_shape: {}, target_name: \"{}\", sensor_data_shape: {}, subject_info_shape: {}"
              .format(dataset.shape, target, sensor_data.shape, subject_info.shape))

        self.subject_info = subject_info
        self.dataset = dataset
        self.target = target
        return dataset, target

    def load_all(self):
        # Unzip
        log.d("Motion sense start unzip")
        self.__unzip()

        # Parse subject info
        log.d("Motion sense start parse")
        subject_info = self.__load_subject_info()
        subject_num = len(subject_info.index)

        # Parse sensor data
        target = self.TRIAL_TYPE_CN
        df = pd.DataFrame()
        for i in range(1, subject_num + 1):
            sensor_data = self.__load_sensor_data(self.TRIALS_NUM * [i])

            # Merge sensor data with subject info
            dataset = sensor_data.merge(subject_info,
                                        left_on=self.SUBJECT_CN,
                                        right_on=subject_info.columns[0])
            # del dataset[self.DATA_SUBJECT_COLUMN_NAME]
            del dataset[subject_info.columns[0]]

            df = df.append(dataset)

        dataset = df

        self.subject_info = subject_info
        self.dataset = dataset
        self.target = target
        return dataset, target

    def print_stats(self):
        if not self.dataset.empty and not self.subject_info.empty and self.target is not None:
            print("************************ Motion Sense - Stats ************************")
            print("[DATASET SHAPE]:        {};".format(self.dataset.shape))
            print("[SUBJECT_INFO SHAPE]:   {};".format(self.subject_info.shape))
            print("[TARGET NAME]:          \'{}\';".format(self.target))
            print("[DATASET COLUMN NAMES]: {};".format(list(self.dataset.columns)))
            print("[TARGET VALUES]:        {};".format(list(set(self.dataset[self.target]))))
            print("[TARGET OCCURRENCES]:\n{};".format(self.dataset[self.target].value_counts()))
            print("[SUBJECT OCCURRENCES]:\n{};".format(self.dataset[self.SUBJECT_CN].value_counts().sort_index()))
            print("[NULL OCCURRENCES]:\n{};".format(self.dataset.isnull().sum()))
            print("**********************************************************************")
        else:
            log.e("Empty dataframe")


class ScooterTrajectoriesDS:
    ZIP_DEFAULT_FN = "scooter_trajectories.zip"
    CSV_DEVICE_FN = "device.csv"
    CSV_POS_FN = ["pos.csv", "pos_1.csv"]
    CSV_RENTAL_FN = "rental.csv"
    CSV_USR_FN = "usr.csv"

    CSV_POS_FILTERED_FN = "pos_filter.csv"
    CSV_RENTAL_FILTERED_FN = "rental_filter.csv"
    CSV_POS_RENTAL_MAP_FN = "pos_rental_map.csv"

    # device.csv data columns
    DEVICE_COLS = ["ID", "Km_tot"]
    DEVICE_ID_CN = DEVICE_COLS[0]
    DEVICE_KM_CN = DEVICE_COLS[1]

    # usr.csv data columns
    USR_COLS = ["ID", "Km_percorsi"]
    USR_ID_CN = USR_COLS[0]
    USR_KM_CN = USR_COLS[1]

    # pos.csv and pos_1.csv data columns
    POS_COLS = ["id", "latitude", "longitude", "speed", "server_time", "device_time", "deviceId"]
    POS_TIME_COLS = [POS_COLS[4], POS_COLS[5]]
    POS_ID_CN = POS_COLS[0]
    POS_LATITUDE_CN = POS_COLS[1]
    POS_LONGITUDE_CN = POS_COLS[2]
    POS_SPEED_CN = POS_COLS[3]
    POS_SERVER_TIME_CN = POS_COLS[4]
    POS_DEVICE_TIME_CN = POS_COLS[5]
    POS_DEVICE_ID_CN = POS_COLS[6]

    # rental.csv data columns
    RENTAL_COLS = ["ID", "Id_dispositivo", "Id_utente", "Start_latitudine", "Start_longitudine", "Stop_latitudine",
                   "Stop_longitudine", "DataOra_partenza", "DataOra_arrivo", "Km_percorsi"]
    RENTAL_TIME_COLS = [RENTAL_COLS[7], RENTAL_COLS[8]]
    RENTAL_ID_CN = RENTAL_COLS[0]
    RENTAL_DEVICE_ID_CN = RENTAL_COLS[1]
    RENTAL_USR_ID_CN = RENTAL_COLS[2]
    RENTAL_START_LATITUDE_CN = RENTAL_COLS[3]
    RENTAL_START_LONGITUDE_CN = RENTAL_COLS[4]
    RENTAL_STOP_LATITUDE_CN = RENTAL_COLS[5]
    RENTAL_STOP_LONGITUDE_CN = RENTAL_COLS[6]
    RENTAL_START_TIME_CN = RENTAL_COLS[7]
    RENTAL_STOP_TIME_CN = RENTAL_COLS[8]
    RENTAL_KM_CN = RENTAL_COLS[9]

    RENTAL_POSITIONS_CN = "Positions"
    POS_RENTAL_CN = "Rental"
    POS_RENTAL_MAP_COLS = ["device_id", "rental_id", 'rental_start_time', 'rental_stop_time', 'pos_id',
                           'pos_device_time', 'pos_server_time']

    def __init__(self, unzip_path=None):
        if unzip_path:
            self.unzip_path = unzip_path
        else:
            self.unzip_path = os.path.join(DATA_FOLDER, self.ZIP_DEFAULT_FN)

        self.unzip_folder = os.path.splitext(self.unzip_path)[0]
        self.dataset = pd.DataFrame()
        self.rental = pd.DataFrame()
        self.pos = pd.DataFrame()

    def __unzip(self):
        if not os.path.exists(self.unzip_path):
            log.f("Unzip path not exist")
            return

        unzip(self.unzip_path)

    def __is_pos_timestamp_in_rental(self, pos, rental):
        pos_server_time_cn = self.POS_TIME_COLS[0]
        pos_device_time_cn = self.POS_TIME_COLS[1]
        rental_start_time_cn = self.RENTAL_TIME_COLS[0]
        rental_end_time_cn = self.RENTAL_TIME_COLS[1]

        return (pos[pos_server_time_cn] >= rental[rental_start_time_cn]) & \
               (pos[pos_device_time_cn] >= rental[rental_start_time_cn]) & \
               (pos[pos_server_time_cn] <= rental[rental_end_time_cn]) & \
               (pos[pos_device_time_cn] <= rental[rental_end_time_cn])

    def __parse_pos_datetime(self, df):
        for date_cn in self.POS_TIME_COLS:
            df[date_cn] = pd.to_datetime(df[date_cn])
        return df

    def __load_device_data(self):
        if not os.path.exists(self.unzip_folder):
            log.e("Scooter Trajectories folder not extracted: impossible to load device data")
            return pd.DataFrame(), get_elapsed(0, 0)

        start = time.time()
        # Parse device data
        device_file = os.path.join(self.unzip_folder, self.CSV_DEVICE_FN)
        device_data = pd.read_csv(device_file, memory_map=True)
        end = time.time()

        del device_data[device_data.columns[0]]
        return device_data, get_elapsed(start, end)

    def __load_pos_data(self, chunknum=None, chunksize=50000, max_chunknum=None):
        if not os.path.exists(self.unzip_folder):
            log.e("Scooter Trajectories folder not extracted: impossible to load position data")
            return pd.DataFrame(), get_elapsed(0, 0)

        start = time.time()
        pos_data = pd.DataFrame()

        # Parse device data
        curr_chunk = 0
        for fn in self.CSV_POS_FN:
            header = 0  # if fn == self.CSV_POS_FN[0] else None
            pos_file = os.path.join(self.unzip_folder, fn)
            reader = pd.read_csv(pos_file, names=self.POS_COLS, header=header, chunksize=chunksize,
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
        rental_file = os.path.join(self.unzip_folder, self.CSV_RENTAL_FN)
        rental_data = pd.read_csv(rental_file, parse_dates=self.RENTAL_TIME_COLS, infer_datetime_format=True,
                                  memory_map=True)

        del rental_data[rental_data.columns[0]]
        end = time.time()
        return rental_data, get_elapsed(start, end)

    def __load_user_data(self):
        if not os.path.exists(self.unzip_folder):
            log.e("Scooter Trajectories folder not extracted: impossible to load user data")
            return pd.DataFrame(), get_elapsed(0, 0)

        start = time.time()
        # Parse device data
        user_file = os.path.join(self.unzip_folder, self.CSV_USR_FN)
        user_data = pd.read_csv(user_file, memory_map=True)

        del user_data[user_data.columns[0]]
        end = time.time()
        return user_data, get_elapsed(start, end)

    def __load(self, device_df, rental_df, user_df, chunknum=None, chunksize=50000, max_chunknum=None):
        if not os.path.exists(self.unzip_folder):
            log.e("Scooter Trajectories folder not extracted: impossible to load pos and rental map data")
            return pd.DataFrame(), get_elapsed(0, 0)

        start = time.time()
        load_data = pd.DataFrame()
        pos_data = pd.DataFrame(columns=self.POS_COLS)

        # Parse device data
        curr_chunk = 0
        for fn in self.CSV_POS_FN:
            header = 0  # if fn == self.CSV_POS_FN[0] else None
            pos_file = os.path.join(self.unzip_folder, fn)
            reader = pd.read_csv(pos_file, names=self.POS_COLS, header=header, chunksize=chunksize, iterator=True,
                                 memory_map=True)

            for pos_chunk_df in reader:
                # Filter pos and rental data
                # pos_chunk_df, rental_df, filter_time = self.__filter(device_df, rental_df, user_df,
                #                                                                  pos_chunk_df)
                # log.d(" - filter elapsed time: {}".format(filter_time))

                # Combine positions to related rental
                # self.dataset = self.__map_pos_into_rental(rental_filtered_df, pos_filtered_df)
                # self.dataset = self.__map_rental_into_pos(rental_filtered_df, pos_filtered_df)
                # pos_rental_map_df, merge_elapsed_time = self.__merge_rental_into_pos(pos_filtered_df,
                #                                                                      rental_filtered_df)
                if chunknum is not None:
                    if chunknum == curr_chunk:
                        pos_chunk_df = self.__parse_pos_datetime(pos_chunk_df)
                        pos_rental_map_df, merge_elapsed_time = self.__merge_rental_into_pos(pos_chunk_df, rental_df)
                        log.d("__chunk {} merge elapsed time: {}".format(curr_chunk, merge_elapsed_time))
                        reader.close()
                        end = time.time()
                        return pos_rental_map_df, pos_chunk_df, get_elapsed(start, end)
                else:
                    # Load all chunk
                    pos_chunk_df = self.__parse_pos_datetime(pos_chunk_df)
                    pos_rental_map_df, merge_elapsed_time = self.__merge_rental_into_pos(pos_chunk_df, rental_df)
                    log.d("__chunk {} merge elapsed time: {}".format(curr_chunk, merge_elapsed_time))
                    load_data = pd.concat([load_data, pos_rental_map_df], axis=0)
                    pos_data = pd.concat([pos_data, pos_chunk_df], axis=0)

                    if max_chunknum is not None and max_chunknum == curr_chunk:
                        end = time.time()
                        return load_data, pos_data, get_elapsed(start, end)

                curr_chunk += 1

            reader.close()

        end = time.time()
        return load_data, pos_data, get_elapsed(start, end)

    def __map_pos_into_rental(self, rental_df: pd.DataFrame, pos_df: pd.DataFrame):
        def ___find_pos_of_rental(p_df, rental, rental_idx):
            start = time.time()
            pos_of_rental = p_df.loc[
                (p_df[self.POS_DEVICE_ID_CN] == rental[self.RENTAL_DEVICE_ID_CN])
                & self.__is_pos_timestamp_in_rental(p_df, rental)]
            end = time.time()
            print("__rental {}; positions found: {}; time: {};"
                  .format(rental_idx, pos_of_rental.index.to_numpy(), get_elapsed(start, end)))
            return pos_of_rental

        rental_df[self.RENTAL_POSITIONS_CN] = [
            ___find_pos_of_rental(pos_df, rental, rental_idx)
            for rental_idx, rental in rental_df.iterrows()]

        # Take only the rental with positions not empty
        not_empty_positions = [pos.size != 0 for pos in rental_df[self.RENTAL_POSITIONS_CN]]
        rental_df = rental_df[not_empty_positions]
        return rental_df

    def __map_rental_into_pos(self, rental_df: pd.DataFrame, pos_df: pd.DataFrame):
        def ___find_rental_of_pos(r_df, pos, pos_idx):
            rental_of_pos = r_df.loc[
                (rental_df[self.RENTAL_DEVICE_ID_CN] == pos[self.POS_DEVICE_ID_CN])
                & self.__is_pos_timestamp_in_rental(pos, r_df)].to_numpy()
            if pos_idx % 1000 == 0:
                log.d("__position {:10} processed".format(pos_idx))
            return rental_of_pos

        pos_df[self.POS_RENTAL_CN] = [___find_rental_of_pos(rental_df, pos, pos_idx)
                                      for pos_idx, pos in pos_df.iterrows()]

        # Remove positions without rentals
        pos_df = pos_df[[rentals.size != 0 for rentals in pos_df[self.POS_RENTAL_CN]]]

        return pos_df

    def __merge_rental_into_pos(self, pos_df: pd.DataFrame, rental_df: pd.DataFrame):
        start = time.time()

        # Merge rental data with pos data
        rental_df = rental_df.rename(columns={self.RENTAL_DEVICE_ID_CN: self.POS_DEVICE_ID_CN})
        rental_merge_cn = [self.POS_DEVICE_ID_CN, self.RENTAL_ID_CN,
                           self.RENTAL_START_TIME_CN, self.RENTAL_STOP_TIME_CN]
        pos_merge_cn = [self.POS_DEVICE_ID_CN, self.POS_ID_CN, self.POS_DEVICE_TIME_CN, self.POS_SERVER_TIME_CN]
        merged_df = rental_df[rental_merge_cn].merge(pos_df[pos_merge_cn], on=self.POS_DEVICE_ID_CN)
        # Filter pos data in rental timestamp range
        timestamp_filtered_df = merged_df.loc[
            self.__is_pos_timestamp_in_rental(merged_df[self.POS_TIME_COLS], merged_df[self.RENTAL_TIME_COLS])]

        end = time.time()
        return timestamp_filtered_df, get_elapsed(start, end)

    def __filter(self, device_df, rental_df, user_df, pos_df):
        start = time.time()
        # Filter rental according to user id and device id
        rental_filtered_df = rental_df.loc[rental_df[self.RENTAL_DEVICE_ID_CN].isin(device_df[self.USR_ID_CN]) &
                                           rental_df[self.RENTAL_USR_ID_CN].isin(user_df[self.DEVICE_ID_CN]) &
                                           rental_df[self.RENTAL_DEVICE_ID_CN].isin(pos_df[self.POS_DEVICE_ID_CN])]

        # Filter positions according to rentals
        pos_filtered_df = pos_df.loc[pos_df[self.POS_DEVICE_ID_CN].isin(rental_filtered_df[self.RENTAL_DEVICE_ID_CN])]

        end = time.time()
        return pos_filtered_df, rental_filtered_df, get_elapsed(start, end)

    def __find_data_map(self, rental_pos_map_df, pos_df, rental_df):
        rental_valid_df = rental_df.loc[rental_df[self.RENTAL_ID_CN].isin(rental_pos_map_df[self.RENTAL_ID_CN])]
        pos_valid_df = pos_df.loc[pos_df[self.POS_ID_CN].isin(rental_pos_map_df[self.POS_ID_CN])]
        return pos_valid_df, rental_valid_df

    def __load_support_data(self):
        start = time.time()
        device_df, _ = self.__load_device_data()
        rental_df, _ = self.__load_rental_data()
        user_df, _ = self.__load_user_data()
        end = time.time()
        return device_df, rental_df, user_df, get_elapsed(start, end)

    def load(self, chunknum=0, chunksize=50000):
        log.d("Scooter Trajectories start unzip")
        self.__unzip()

        log.d("Scooter Trajectories load device, rental, user data")
        device_df, rental_df, user_df, support_data_load_time = self.__load_support_data()
        log.d("elapsed time: {}".format(support_data_load_time))

        # Load dataset and positions according to chunk info
        map_pos_rental_df, pos_df, load_time = self.__load(device_df, rental_df, user_df, chunknum, chunksize)
        log.d(" - load map elapsed time: {}".format(load_time))

        # Calculate valid positions and rental according to the each other timestamp mapping
        self.pos, self.rental = self.__find_data_map(map_pos_rental_df, pos_df, rental_df)

        # Rename dataset columns.
        self.dataset = map_pos_rental_df.rename(columns=dict(zip(map_pos_rental_df.columns, self.POS_RENTAL_MAP_COLS)))

        return self

    def load_all(self, chunksize=50000, max_chunknum=None):
        log.d("Scooter Trajectories start unzip")
        self.__unzip()

        log.d("Scooter Trajectories load device, rental, user data")
        device_df, rental_df, user_df, support_data_load_time = self.__load_support_data()
        log.d("elapsed time: {}".format(support_data_load_time))

        log.d("Scooter Trajectories load pos and rental timestamp map data")
        map_pos_rental_df, pos_df, load_time = self.__load(device_df, rental_df, user_df, chunknum=None,
                                                           chunksize=chunksize, max_chunknum=max_chunknum)
        log.d("elapsed time: {}".format(load_time))

        # Calculate valid positions and rental according to the each other timestamp mapping
        self.pos, self.rental = self.__find_data_map(map_pos_rental_df, pos_df, rental_df)

        # Rename dataset columns.
        self.dataset = map_pos_rental_df.rename(columns=dict(zip(map_pos_rental_df.columns, self.POS_RENTAL_MAP_COLS)))

        return self

    def to_csv(self):
        if self.dataset.empty or self.rental.empty or self.pos.empty:
            log.e("Empty dataframe: to_csv() error")
            return self

        self.pos.to_csv(os.path.join(DATA_FOLDER, self.CSV_POS_FILTERED_FN), index=False)
        self.rental.to_csv(os.path.join(DATA_FOLDER, self.CSV_RENTAL_FN), index=False)
        self.dataset.to_csv(os.path.join(DATA_FOLDER, self.CSV_POS_RENTAL_MAP_FN), index=False)
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
        log.i("[DATA MAP DESCRIPTION]:\n{};".format(self.dataset.describe(datetime_is_numeric=True)))
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

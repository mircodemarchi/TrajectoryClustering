import pandas as pd
import numpy as np
import os

from util.log import Log
from util.constant import DATA_FOLDER
from util.util import unzip

from .constant import MotionSenseC as C

log = Log(__name__, enable_console=True, enable_file=False)


class MotionSenseDS:
    def __init__(self, unzip_path=None, log_lvl=None):
        if unzip_path:
            self.unzip_path = unzip_path
        else:
            self.unzip_path = os.path.join(DATA_FOLDER, C.ZIP_DEFAULT_FN)

        if log_lvl is not None:
            if not isinstance(log_lvl, int):
                raise ValueError("Invalid log level: {}".format(log_lvl))
            log.set_level(log_lvl)

        self.unzip_folder = os.path.splitext(self.unzip_path)[0]
        self.target = None
        self.dataset = pd.DataFrame()
        self.subject_info = pd.DataFrame()

    def __is_trial_long(self, trial_number):
        if trial_number in C.TRIALS_LONG:
            return True

        if trial_number in C.TRIALS_SHORT:
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
        subject_info_file = os.path.join(self.unzip_folder, C.CSV_SUBJECT_INFO_FN)
        subject_info_data = pd.read_csv(subject_info_file)

        return subject_info_data

    def __load_sensor_data(self, mask):
        data_subfolder = os.path.join(self.unzip_folder, C.SUBDIRNAME)
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

                    data_trial[C.TRIAL_LONG_CN] = np.full(len(data_trial.index),
                                                          self.__is_trial_long(current_trial_num))
                    data_trial[C.SUBJECT_CN] = np.full(len(data_trial.index), selected_subject)
                    data_trial[C.TRIAL_TYPE_CN] = np.full(len(data_trial.index), current_trial_type)

                    df = df.append(data_trial)

        return df

    def load(self, mask=None):
        if mask is None or mask.shape[0] != C.TRIALS_NUM:
            mask = C.TRIALS_NUM * [0]

        # Unzip
        log.d("Motion sense start unzip")
        self.__unzip()

        # Parse subject info
        log.d("Motion sense start parse")
        subject_info = self.__load_subject_info()

        # Parse sensor data
        target = C.TRIAL_TYPE_CN
        sensor_data = self.__load_sensor_data(mask)

        # Merge sensor data with subject info
        if not sensor_data.empty:
            dataset = sensor_data.merge(subject_info,
                                        left_on=C.SUBJECT_CN,
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
        target = C.TRIAL_TYPE_CN
        df = pd.DataFrame()
        for i in range(1, subject_num + 1):
            sensor_data = self.__load_sensor_data(C.TRIALS_NUM * [i])

            # Merge sensor data with subject info
            dataset = sensor_data.merge(subject_info,
                                        left_on=C.SUBJECT_CN,
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
            print("[SUBJECT OCCURRENCES]:\n{};".format(self.dataset[C.SUBJECT_CN].value_counts().sort_index()))
            print("[NULL OCCURRENCES]:\n{};".format(self.dataset.isnull().sum()))
            print("**********************************************************************")
        else:
            log.e("Empty dataframe")
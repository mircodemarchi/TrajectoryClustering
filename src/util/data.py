import pandas as pd
import numpy as np
import zipfile
import os
import sys
from tqdm import tqdm
from util import Log

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
    TRIALS_LONG = range(1, 10) # [1, 9]
    TRIALS_SHORT = range(11, 17)  # [11, 16]

    DATA_ZIP_DEFAULT_FILENAME = "motion-sense.zip"
    DATA_DIRNAME = "A_DeviceMotion_data"
    DATA_CSV_SUBJECT_INFO_FILENAME = "data_subjects_info.csv"
    DATA_TRIAL_LONG_COLUMN_NAME = "is long"
    DATA_SUBJECT_COLUMN_NAME = "subject"
    DATA_TRIAL_TYPE_COLUMN_NAME = "trial type"

    def __init__(self, unzip_path=None):
        if unzip_path:
            self.unzip_path = unzip_path
        else:
            self.unzip_path = os.path.join(DATA_FOLDER, self.DATA_ZIP_DEFAULT_FILENAME)

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
        if os.path.exists(self.unzip_path):
            log.f("Unzip path not exist")

        unzip(self.unzip_path)

    def __load_subject_info(self):
        if not os.path.exists(self.unzip_folder):
            log.e("Motion sense folder not extracted: impossible to load subject info")
            return

        # Parse subject info
        subject_info_file = os.path.join(self.unzip_folder, self.DATA_CSV_SUBJECT_INFO_FILENAME)
        subject_info_data = pd.read_csv(subject_info_file)

        return subject_info_data

    def __load_sensor_data(self, mask):
        data_subfolder = os.path.join(self.unzip_folder, self.DATA_DIRNAME)
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

                    data_trial[self.DATA_TRIAL_LONG_COLUMN_NAME] = np.full(len(data_trial.index),
                                                                           self.__is_trial_long(current_trial_num))
                    data_trial[self.DATA_SUBJECT_COLUMN_NAME] = np.full(len(data_trial.index), selected_subject)
                    data_trial[self.DATA_TRIAL_TYPE_COLUMN_NAME] = np.full(len(data_trial.index), current_trial_type)

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
        target = self.DATA_TRIAL_TYPE_COLUMN_NAME
        sensor_data = self.__load_sensor_data(mask)

        # Merge sensor data with subject info
        if not sensor_data.empty:
            dataset = sensor_data.merge(subject_info,
                               left_on=self.DATA_SUBJECT_COLUMN_NAME,
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
        target = self.DATA_TRIAL_TYPE_COLUMN_NAME
        df = pd.DataFrame()
        for i in range(1, subject_num+1):
            sensor_data = self.__load_sensor_data(self.TRIALS_NUM * [i])

            # Merge sensor data with subject info
            dataset = sensor_data.merge(subject_info,
                                        left_on=self.DATA_SUBJECT_COLUMN_NAME,
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
            print("[SUBJECT OCCURRENCES]:\n{};".format(self.dataset[self.DATA_SUBJECT_COLUMN_NAME].value_counts().sort_index()))
            print("[NULL OCCURRENCES]:\n{};".format(self.dataset.isnull().sum()))
            print("**********************************************************************")
        else:
            log.e("Empty dataframe")




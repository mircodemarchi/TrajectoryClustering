import pandas as pd
import numpy as np
import zipfile
import os
import random
import sys
from tqdm import tqdm
from util import Log

DATA_FOLDER = "../data"

DATA_ZIP_MOTION_SENSE = "motion-sense.zip"

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


def unzip(name):
    path = os.path.join(DATA_FOLDER, name)
    with zipfile.ZipFile(path, "r") as zf:

        # If unzip folder already exist, check same size
        target_dir = os.path.splitext(path)[0]
        if os.path.exists(target_dir):
            uncompress_size = sum((file.file_size for file in zf.infolist()))
            if get_folder_size(target_dir) == uncompress_size:
                gen = tqdm(desc=name, total=len(zf.namelist()), file=sys.stdout)
                gen.update(len(zf.namelist()))
                gen.close()
                return

        # Extract
        gen = tqdm(desc=name, iterable=zf.namelist(), total=len(zf.namelist()), file=sys.stdout)
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


def load_motion_sense(mask=None):
    if mask is None:
        mask = 15 * [0]

    # Unzip
    log.d("Motion sense start unzip")
    unzip(DATA_ZIP_MOTION_SENSE)
    unzip_folder = os.path.join(DATA_FOLDER, os.path.splitext(DATA_ZIP_MOTION_SENSE)[0])

    # Parse subject info
    log.d("Motion sense start parse")
    subject_info_file = os.path.join(unzip_folder, "data_subjects_info.csv")
    subject_info_data = pd.read_csv(subject_info_file)

    # Parse data
    long_trials = range(1, 10) # [1, 9]
    short_trials = range(11, 17)  # [11, 16]

    def is_trial_long(trial_number):
        if trial_number in long_trials:
            return True

        if trial_number in short_trials:
            return False

        return None

    data_subfolder = os.path.join(unzip_folder, "A_DeviceMotion_data")
    target = "Trial Type"
    data = pd.DataFrame()
    i = 0
    for dirpath, dirnames, filenames in os.walk(data_subfolder):
        if data_subfolder != dirpath:
            # Extract trial type from the current dirname.
            trial_type = os.path.basename(dirpath).split("_")[0]

            # Extract trial number from the current dirname.
            trial_num = int(os.path.basename(dirpath).split("_")[1])

            # Extract data of a random subject for the current trial.
            random_subject_trial_file = filenames[mask[i]]
            random_subject = int(os.path.splitext(random_subject_trial_file)[0].split("_")[1])
            data_trial = pd.read_csv(os.path.join(dirpath, random_subject_trial_file))
            del data_trial[data_trial.columns[0]]

            data_trial["Is Long"] = np.full(len(data_trial.index), is_trial_long(trial_num))
            data_trial["Subject"] = np.full(len(data_trial.index), random_subject)
            data_trial[target] = np.full(len(data_trial.index), trial_type)

            data = data.append(data_trial)

    dataset = data.merge(subject_info_data, left_on="Subject", right_on=subject_info_data.columns[0])
    del dataset["Subject"]
    del dataset[subject_info_data.columns[0]]

    log.i("Motion sense dataset_shape: {}, target_name: \"{}\", sensor_data_shape: {}, subject_info_shape: {}"
          .format(dataset.shape, target, data.shape, subject_info_data.shape))

    return dataset, target


from typing import Final


class MotionSenseC:
    TRIALS_NUM: Final = 15
    TRIALS_LONG: Final = range(1, 10)  # [1, 9]
    TRIALS_SHORT: Final = range(11, 17)  # [11, 16]

    ZIP_DEFAULT_FN: Final = "motion-sense.zip"
    SUBDIRNAME: Final = "A_DeviceMotion_data"
    CSV_SUBJECT_INFO_FN: Final = "data_subjects_info.csv"
    TRIAL_LONG_CN: Final = "is long"
    TRIAL_TYPE_CN: Final = "trial type"
    SUBJECT_CN: Final = "subject"

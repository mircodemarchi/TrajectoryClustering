
import sys
import os
import zipfile
import tqdm

from .constant import DATA_FOLDER


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


def get_elapsed(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
import os


def create_dirs(dirs):

    try:
        for _dir in dirs:
            if not os.path.exists(_dir):
                os.makedirs(_dir)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)

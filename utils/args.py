import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-c",
        "--config",
        dest="config",
        metavar="C",
        default="None",
        help="The Configuration file"
    )
    args = argparse.parse_args()
    return args

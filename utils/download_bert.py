import os
import sys
import argparse
import urllib.request
import zipfile

_BERT_LARGE_URL = 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip'
_BERT_BASE_URL = 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip'

_BERT_LARGE_DIR = 'uncased_L-24_H-1024_A-16'
_BERT_BASE_DIR = 'uncased_L-12_H-768_A-12'


def download_base(path):
    print("Downloading and extracting bert base")
    data_file = "%s.zip" % _BERT_BASE_DIR
    urllib.request.urlretrieve(_BERT_BASE_URL, data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(path)
    os.remove(data_file)
    print("\tCompleted!")


def download_large(path):
    print("Downloading and extracting bert large")
    data_file = "%s.zip" % _BERT_LARGE_DIR
    urllib.request.urlretrieve(_BERT_LARGE_URL, data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(path)
    os.remove(data_file)
    print("\tCompleted!")


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bert_dir', help='directory to save data to', type=str, default='./data')
    parser.add_argument('--bert_model', help='specify base or large uncased bert model',
                        type=str, default='base')
    args = parser.parse_args(arguments)

    if not os.path.isdir(args.bert_dir):
        os.mkdir(args.bert_dir)

    assert args.bert_model in ['base', 'large',
                               'both'], "model not specified properly"

    if args.bert_model == 'base':
        download_base(args.bert_dir)
    elif args.bert_model == 'large':
        download_large(args.bert_dir)
    else:
        download_base(args.bert_dir)
        download_large(args.bert_dir)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

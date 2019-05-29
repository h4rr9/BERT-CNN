import os
import sys
import codecs
import argparse


def parse_tasks(path):
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.tsv'):
                print('cleaning {0}'.format(os.path.join(subdir, file)))

                with codecs.open(os.path.join(subdir, file), 'r', encoding='utf8') as _file:
                    content = _file.readlines()

                formated = []
                for line in content:
                    formated.append(line.replace('"', ''))

                with codecs.open(os.path.join(subdir, file), 'w', encoding='utf8') as _file:
                    _file.write(''.join(formated))


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='data directory',
                        type=str, default='../data/glue_data')

    args = parser.parse_args(arguments)

    if not os.path.exists(args.data_dir):
        raise Exception("PathNotFound")

    parse_tasks(args.data_dir)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

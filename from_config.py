from utils.config import process_config
from utils.dirs import create_dirs
from utils.args import get_args
from utils import factory
import sys
import os
import warnings

warnings.simplefilter("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'


def main():

    try:
        args = get_args()
        config = process_config(args.config)

        create_dirs([config.callbacks.tensorboard_log_dir,
                     config.callbacks.checkpoint_dir])

        print("Create the data generator.")
        data_loader = factory.create(
            "data_loader." + config.data_loader.name)(config)

        print("Create the model.")
        model = factory.create("models." + config.model.name)(config)

        print("Create the trainer.")
        trainer = factory.create(
            "trainers." + config.trainer.name)(model.model, data_loader, config)

        print("Start training the model.")
        trainer.train()

    except Exception as e:
        print(e.with_traceback())
        sys.exit(-1)


if __name__ == "__main__":
    main()

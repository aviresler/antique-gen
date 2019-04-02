from utils.config import process_config
from utils.dirs import create_dirs
from utils.args import get_args
from utils import factory
from data_loader.default_generator import get_default_generator
from models.cosloss_model import CosLosModel
from trainers.cosloss_trainer import CosLossModelTrainer
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    # capture the config path from the run arguments
    # then process the json configuration fill
    try:
        args = get_args()
        config = process_config(args.config)

        # create the experiments dirs
        create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

        print('train generator')
        train_generator = get_default_generator(config, True)
        valid_generator = get_default_generator(config, False)

        print('Create the model.')
        model = CosLosModel(config)

        print('Create the trainer')
        trainer = CosLossModelTrainer(model.model, train_generator ,valid_generator, config)

        print('Start training the model.')
        trainer.train()

    except Exception as e:
        print(e)
        sys.exit(1)

if __name__ == '__main__':
    main()

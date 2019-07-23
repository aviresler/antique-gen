from keras.preprocessing.image import ImageDataGenerator
from data_loader.preprocess_images import get_random_eraser, preprocess_input

def get_default_generator(config, is_train = True):
    if config.data_loader.is_use_cutOut:
        preprocess_function = get_random_eraser(v_l=0, v_h=1)
    else:
        preprocess_function = preprocess_input


    if is_train:
        data_dir = config.data_loader.data_dir_train
    else:
        data_dir = config.data_loader.data_dir_valid

    train_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_function,
    horizontal_flip = config.data_loader.horizontal_flip,
    fill_mode = config.data_loader.fill_mode,
    zoom_range = config.data_loader.zoom_range,
    width_shift_range = config.data_loader.width_shift_range,
    height_shift_range=config.data_loader.height_shift_range,
    rotation_range=config.data_loader.rotation_range)

    train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size= (config.model.img_height, config.model.img_width),
    batch_size= config.data_loader.batch_size,
    class_mode= config.data_loader.class_mode)

    return train_generator


def get_testing_generator(config, is_train = False, folder=''):

    if folder == '':
        if is_train:
            data_dir = config.data_loader.data_dir_train_test
        else:
            data_dir = config.data_loader.data_dir_valid_test
    else:
        data_dir = folder


    test_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    horizontal_flip = False,
    vertical_flip= False)

    test_generator = test_datagen.flow_from_directory(
    data_dir,
    target_size= (config.model.img_height, config.model.img_width),
    batch_size= config.data_loader.batch_size,
    class_mode= "sparse",
    shuffle=False)

    return test_generator
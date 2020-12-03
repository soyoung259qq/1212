from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def adjust_data(img, mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    return (img, mask)


def train_generator(data_frame, batch_size, aug_dict,
                    image_color_mode="rgb",
                    mask_color_mode="grayscale",
                    image_save_prefix="image",
                    mask_save_prefix="mask",
                    save_to_dir=None,
                    target_size=(256, 256),
                    seed=1):
    '''
    can generate image and mask at the same time use the same seed for
    image_datagen and mask_datagen to ensure the transformation for image
    and mask is the same if you want to visualize the results of generator,
    set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        x_col="filename",
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)

    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        x_col="mask",
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    train_gen = zip(image_generator, mask_generator)

    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        yield (img, mask)


def load_train_data(path):
    train_files = []
    mask_files = glob(f'{path}/*/*_mask*')

    for i in mask_files:
        train_files.append(i.replace('_mask',''))

    print(train_files[:10])
    print(mask_files[:10])

    df = pd.DataFrame(data={"filename": train_files, 'mask' : mask_files})
    df_train, df_test = train_test_split(df,test_size = 0.1)
    df_train, df_val = train_test_split(df_train,test_size = 0.2)
    print(df_train.values.shape)
    print(df_val.values.shape)
    print(df_test.values.shape)

    return df_train, df_val, df_test


def get_train_generator(path, batch_size, target_size):
    df_train, df_val, df_test = load_train_data(path)

    train_generator_args = dict(rotation_range=0.2,
                                width_shift_range=0.05,
                                height_shift_range=0.05,
                                shear_range=0.05,
                                zoom_range=0.05,
                                horizontal_flip=True,
                                fill_mode='nearest')

    train_gen = train_generator(df_train, batch_size,
                                train_generator_args,
                                target_size=target_size)

    valid_gen = train_generator(df_val, batch_size,
                                 dict(),
                                 target_size=target_size)

    test_gen = train_generator(df_test, batch_size,
                                 dict(),
                                 target_size=target_size)

    return train_gen, valid_gen, test_gen, df_train, df_val

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from Data import Data
from Model import Model
from Train import Train
from Test import Test
from Utils import Drawer


def train(train_gen, valid_gen, df_train, df_val, batch_size, target_size):
    epochs = 150
    Drawer.draw_data_samples(df_train)
    unet_model = Model.create_model(input_size=target_size + (3,))
    Train.run_train(unet_model, train_gen=train_gen, valid_gen=valid_gen, batch_size=batch_size,
                    df_train=df_train, df_val=df_val, epochs=epochs)


def test(test_gen, model_path):
    test_result = Test.run_test(test_gen, model_path, 10)
    Drawer.draw_test_result(test_result)



if __name__ == '__main__':

    target_size = (256, 256)
    batch_size = 2 # tune depends on graphic card memory card
    path = r"D:\Dataset\MRI\lgg-mri-segmentation\kaggle_3m"
    train_gen, valid_gen, test_gen, df_train, df_val = Data.get_train_generator(batch_size=batch_size,
                                                                                target_size=target_size, path=path)

    mode = "train"

    if mode is "train":
        train(train_gen, valid_gen, df_train, df_val, batch_size, target_size)
    else:
        test(test_gen, "./unet_brain_mri_seg2.hdf5")




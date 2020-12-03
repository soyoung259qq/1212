from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from Losses.Losses import dice_coef_loss, iou, dice_coef



def run_train(model, train_gen, df_train, valid_gen, df_val, batch_size, epochs):
    learning_rate = 1e-4
    decay_rate = learning_rate / epochs
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)
    model.compile(optimizer=opt, loss=dice_coef_loss, metrics=["binary_accuracy", iou, dice_coef])

    callbacks = [ModelCheckpoint('unet_brain_mri_seg.hdf5', verbose=1, save_best_only=True)]

    history = model.fit(train_gen,
                        steps_per_epoch= len(df_train) / batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=valid_gen,
                        validation_steps=len(df_val) / batch_size)
    return history

from tensorflow.keras.models import Model, load_model
from Losses.Losses import dice_coef_loss, iou, dice_coef
import numpy as np

def run_test(test_gen, model_path, test_size=-1):

    model = load_model(model_path, custom_objects={"dice_coef_loss":dice_coef_loss, "iou":iou, "dice_coef":dice_coef})
    result = list()
    cnt = 0
    if test_size < 0:
        predict = model.predict(test_gen)
        result.append((predict, test_gen))
        return result

    for samples in test_gen:
        for img, gt in zip(samples[0],samples[1]):
            predict = model.predict(np.reshape(img, (1,)+img.shape))
            result.append((img, np.reshape(predict, predict.shape[1:]), gt))
            cnt += 1

            if cnt >= test_size:
                return result


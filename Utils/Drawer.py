import matplotlib.pyplot as plt
import cv2


def adjust_data(img, mask):
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return (img, mask)

def draw_data_samples(df_data):

    rows, cols = 3, 3
    fig = plt.figure(figsize=(10, 10))
    for i in range(1, rows * cols + 1):
        fig.add_subplot(rows, cols, i)
        img_path = df_data.iloc[i][0]
        msk_path = df_data.iloc[i][1]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        msk = cv2.imread(msk_path)
        plt.imshow(img)
        plt.imshow(msk, alpha=0.4)
    plt.show()


def draw_test_result(test_result):
    rows, cols = 3, 3
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Predict")
    for i in range(1, rows * cols + 1):
        fig.add_subplot(rows, cols, i)
        img = test_result[i-1][0]
        msk = test_result[i-1][1]
        img, msk = adjust_data(img, msk)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.imshow(msk, alpha=0.4)


    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("GT")
    for i in range(1, rows * cols + 1):
        fig.add_subplot(rows, cols, i)
        img = test_result[i-1][0]
        msk = test_result[i-1][2]
        img, msk = adjust_data(img, msk)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.imshow(msk, alpha=0.4)
    plt.show()
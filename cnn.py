import os   # 用於讀取dataset
import cv2  # 影像處理用
import numpy as np  # 矩陣運算
from random import shuffle  # 用於打亂資料順序
from tensorflow.keras.models import Sequential   # 引入序列模型
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization  # cnn所需的各個層
#from tensorflow.keras.utils import np_utils   # 用於將輸出變為 one hot vector
from tensorflow.python.keras.utils.np_utils import to_categorical
#from keras.optimizers import Adam  # 優化器
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
import matplotlib.pyplot as plt    # 繪圖用
import gc
from datetime import datetime
import tensorflow as tf

def alphabet2num(a):    # 將英文轉數字，即y的數值
    alphabet = 'ABCDEFGHIKLMNOPQRSTUVWXY'
    num = 0
    for j in alphabet:
        if a == j:
            return num
        else:
            num = num+1

def train_model(x_test, y_test, x_train, y_train):
    model = Sequential()  # 序列模型
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))  # 卷積層   32
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))  # 卷積層   64
    model.add(Dropout(0.5))
    model.add(MaxPool2D(pool_size=(2, 2)))  # 池化層
    #
    
    # model.add(MaxPool2D(pool_size=(2, 2)))  # 池化層
    # model.add(Dropout(0.5))  # 防止過擬合
    #
    # model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))  # 卷積層  128
    # model.add(Dropout(0.4))
    # model.add(MaxPool2D(pool_size=(2, 2)))  # 池化層
    # fully connected
    model.add(Flatten())  # 將資料壓平
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(24, activation='softmax'))
    model.summary()
    epochs = 20  # 訓練次數
    batch_size = 16  # 批次
    optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)  # 定義優化器
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    # 訓練模型
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=epochs,
                        batch_size=batch_size)  # validation_data=(x_test, y_test),
    model.save('cnn3.h5')  # 儲存模型
    scores = model.evaluate(x_test, y_test, verbose=0)  # 計算模型正確率
    print("{}: {:.2f}%".format("accuracy", scores[1] * 100))


if __name__ == '__main__':
    y_test = []  # 測試集的正確答案
    x_test = []  # 測試集的輸入圖片(矩陣)
    val_acc = []
    val_loss = []
    test_acc = []
    test_loss = []
    train_acc = []
    train_loss = []

    log_dir = "./logs"

    # 创建 SummaryWriter
    writer = tf.summary.create_file_writer(log_dir)

    # 定义回调函数
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # test data
    test_dataset = os.listdir(r'./his_test_dataset')

    shuffle(test_dataset)  # 打亂順序
    i = 0
    for photo in test_dataset:  # 讀取測試集中每張圖片
        y_test.append(alphabet2num(photo[0]))  # 測試集答案陣列中放入正確答案
        img = cv2.imread('./his_test_dataset/' + photo, cv2.IMREAD_GRAYSCALE)
        img = img.reshape(128, 128, 1)
        x_test.append(img)  # 放入測試集陣列中
        i = i + 1
        if i == 1000:
            break

    x_test = np.array(x_test, dtype='float')  # 轉為numpy array
    x_test = x_test / 255      # normalization
    y_test = to_categorical(y_test)  # one-hot 編碼輸出變量

    val_x = x_test[0:500]
    val_y = y_test[0:500]
    x_test = x_test[500:1000]
    y_test = y_test[500:1000]

    print('x_test_shape: ', x_test.shape)  # 印出測試集大小
    print('y_test_shape: ', y_test.shape)  # 印出測試集大案大小
    print('val_x_shape: ', val_x.shape)  # 印出測試集大小
    print('val_y_shape: ', val_y.shape)  # 印出測試集大案大小


    # train data
    train_dataset = os.listdir(r'./his_train_dataset')
    shuffle(train_dataset)           # 打亂訓練資料順序
    train_name = []                  # 用來儲存所有圖片檔名
    train_label = []                 # 用來儲存所有測試級的正確答案
    for photo2 in train_dataset:     # 讀取訓練集每張圖片
        train_name.append(photo2)    # 儲存所有圖片檔名
        train_label.append(alphabet2num(photo2[0]))  # 儲存所有測試級的正確答案
    train_label = to_categorical(train_label)  # one-hot 編碼輸出變量
    print(f'檔案總數 : {len(train_name)}')


    datagen = ImageDataGenerator(           # 資料隨機做平移或旋轉
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=10, verbose=1, factor=0.8,
                                                min_lr=0.00001)
    callbacks_list = [tensorboard_callback, learning_rate_reduction]
    j = 0
    end = 0
    while 1:
        print(j)
        y_train = []  # 訓練集的正確答案
        x_train = []  # 訓練集的輸入圖片(矩陣)
        # print(len(train_name))
        start = j
        for i in range(1000):   # 載入800張照片
            if j < len(train_name):   # len(train_name)
                img2 = cv2.imread('./his_train_dataset/' + train_name[j], cv2.IMREAD_GRAYSCALE)
                # print(f'img2={img2.shape}')
                img2 = img2.reshape(128, 128, 1)
                x_train.append(img2)  # 放入訓練集陣列中
                j = j+1
            else:
                end = 1
                break
        if j == len(train_name):
            end = 1
        x_train = np.array(x_train, dtype='float')  # 轉成numpy的array
        y_train = train_label[start:j]  # 抓取對應的label

        print('x_train_shape: ', x_train.shape)
        print('y_train_shape: ', y_train.shape)
        # cv2.imshow('train_img', x_train[10].reshape(128, 128))
        # print('y_train[10]: ', y_train[10])
        x_train = x_train / 255  # normalization
        datagen.fit(x_train)
        if j < 1001:              # 第一次訓練，設計模型
            train_model(x_test, y_test, x_train, y_train)
        else:
            epochs = 20  # 訓練次數
            batch_size = 8  # 批次
            model = load_model('cnn3.h5')
            history = model.fit(x_train, y_train,  epochs=epochs, validation_data=(val_x, val_y),
                                batch_size=batch_size, callbacks=[learning_rate_reduction])
            model.save('cnn3.h5')  # 儲存模型
            scores = model.evaluate(x_test, y_test, verbose=0)  # 計算模型正確率
            print("{}: {:.2f}%".format("accuracy", scores[1] * 100))
            val_loss.append(history.history['val_loss'][-1])
            val_acc.append(history.history['val_accuracy'][-1])
            train_loss.append(history.history['loss'][-1])
            train_acc.append(history.history['accuracy'][-1])
            test_loss.append(scores[0])
            test_acc.append(scores[1])
            # with writer.as_default():
                # tf.summary.merge_all()
        gc.collect(generation=2)
        del x_train
        del y_train
        # del model
        if end:
            writer.close()
            break
    plt.figure()
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.plot(test_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'validation_accuracy', 'test_accuracy'], loc='upper right')

    plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.plot(test_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'validation_loss', 'test_loss'], loc='upper right')
    plt.show()
    cv2.waitKey(0)


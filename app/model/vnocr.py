import numpy as np
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Activation, BatchNormalization, Add, Lambda, Bidirectional, LSTM, Dense
from tensorflow.keras import backend as K
import os
import requests
import io
import gdown

import os
import gdown
import io

def download_weights_to_memory():
    # Đường dẫn file weights local
    file_path = "checkpoint_weights.weights.h5"

    # Kiểm tra nếu file đã tồn tại
    if os.path.exists(file_path):
        print(f"File {file_path} đã tồn tại. Không cần tải lại.")
        # Nếu file tồn tại, đọc vào BytesIO
        with open(file_path, 'rb') as f:
            file_stream = io.BytesIO(f.read())
        return file_stream

    # Nếu file không tồn tại, tải từ Google Drive
    FILE_ID = "1cuDkY9y8k7zZV778xqGpLvsFsKdlvkQ6"  # Thay bằng FILE_ID thực tế
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    print("Downloading weights from Google Drive...")
    file_path = gdown.download(url, quiet=False, fuzzy=True)

    # Đọc nội dung file vào BytesIO
    with open(file_path, 'rb') as f:
        file_stream = io.BytesIO(f.read())
    print("Weights downloaded to memory.")

    return file_stream

def load_weights_from_memory(model, file_stream):
    # Tạo file tạm để lưu weights
    temp_file_path = "temp.weights.h5"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file_stream.getbuffer())
    print(f"Temporary weights file created at: {temp_file_path}")

    # Load weights từ file tạm
    model.load_weights(temp_file_path)
    print("Model weights loaded successfully!")

    # Xóa file tạm
    os.remove(temp_file_path)
    print("Temporary weights file removed.")


def load_model(model_folder):
    inputs = Input(shape=(118, 2167, 1))
    
    # Block 1
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = MaxPool2D(pool_size=3, strides=3)(x)
    x = Activation('relu')(x)
    x_1 = x

    # Block 2
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = MaxPool2D(pool_size=3, strides=3)(x)
    x = Activation('relu')(x)
    x_2 = x

    # Block 3
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x_3 = x

    # Block 4
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_3])
    x = Activation('relu')(x)
    x_4 = x

    # Block 5
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x_5 = x

    # Block 6
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_5])
    x = Activation('relu')(x)

    # Block 7
    x = Conv2D(1024, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(3, 1))(x)
    x = Activation('relu')(x)

    # Pooling layer
    x = MaxPool2D(pool_size=(3, 1))(x)

    # Squeeze layer
    squeezed = Lambda(lambda x: K.squeeze(x, 1))(x)

    # BLSTM layers
    blstm_1 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.2))(squeezed)
    blstm_2 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.2))(blstm_1)

    # Load character list
    os.path.join(model_folder, 'char_list.pkl')
    with open(os.path.join(model_folder, 'char_list.pkl'), 'rb') as f:
        char_list = pickle.load(f)
    # print(char_list)
    # Output layer
    outputs = Dense(len(char_list) + 1, activation='softmax')(blstm_2)

    # Define model
    model = Model(inputs, outputs)
    file_stream = download_weights_to_memory()
    load_weights_from_memory(model, file_stream)
    # model.load_weights(os.path.join(model_folder, 'checkpoint_weights.weights.h5'))

    return model, char_list


def predict_line(model, char_list, image):
    prediction = model.predict(image)
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1], greedy=True)[0][0])

    # print("Dự đoán = ", end='')
    pred = ""
    for p in out[0]:
        if int(p) != -1:
            pred += char_list[int(p)]
    # print(pred)

    return pred
# import os
# print("Current working directory:", os.getcwd())
# char_list_path = os.path.join('model', 'char_list.pkl')
# with open(char_list_path, 'rb') as f:
#     char_list = pickle.load(f)

# print(char_list)
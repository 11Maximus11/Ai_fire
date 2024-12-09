# ������ ����������� ���������
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# ������� ���� � ������ � �������������
fire_path = r"C:\Users\myton\Downloads\archive (1)\fire_dataset\fire_images"       # ����� � ������������� ������
non_fire_path = r"C:\Users\myton\Downloads\archive (1)\fire_dataset\non_fire_images"  # ����� � ������������� ��� ������

# ���������, ���� �� ���������� ������
model_path = "fire_detection_model.h5"

if not os.path.exists(model_path):
    print("������ �� �������, ����� ��������...")

    # �������� DataFrame
    data = []  # ��������� ������ ��� �������� ������

    # ���� ������ � �������
    for dirname, _, filenames in os.walk(fire_path):
        for filename in filenames:
            data.append({'path': os.path.join(dirname, filename), 'label': 'fire'})

    # ���� ������ � ������� ��� ������
    for dirname, _, filenames in os.walk(non_fire_path):
        for filename in filenames:
            data.append({'path': os.path.join(dirname, filename), 'label': 'non_fire'})

    # �������������� ������ � DataFrame
    df = pd.DataFrame(data)

    # ������������ ������
    df = df.sample(frac=1).reset_index(drop=True)

    # ������ ������
    sns.countplot(data=df, x='label')
    plt.title("���������� ����������� �� �������")
    plt.show()

    # �������� ����������� ������
    generator = ImageDataGenerator(
        rescale=1/255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        validation_split=0.2
    )

    train_gen = generator.flow_from_dataframe(
        df,
        x_col='path',
        y_col='label',
        target_size=(256, 256),
        class_mode='binary',
        subset='training'
    )

    val_gen = generator.flow_from_dataframe(
        df,
        x_col='path',
        y_col='label',
        target_size=(256, 256),
        class_mode='binary',
        subset='validation'
    )

    # ������ ����� �������
    class_indices = train_gen.class_indices
    print(f"����� �������: {class_indices}")  # {'fire': 0, 'non_fire': 1}

    # �������� ������
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        MaxPool2D(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D(),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # ���������� ������
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # �������� ������
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=15,
        callbacks=[early_stopping, reduce_lr]
    )

    # ���������� ������
    model.save(model_path)
    print("������ ���������!")
else:
    # �������� ���������� ������
    model = load_model(model_path)
    print("������ ���������!")

# ������������ ������ �� ����� ������������
def predict_image(image_path):
    # �������� �����������
    img = load_img(image_path, target_size=(256, 256))  # ����������� ������
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ������������
    prediction = model.predict(img_array)
    label = 'fire' if prediction[0][0] < 0.5 else 'non_fire'
    print(f"����������� {image_path}: {label}")

# ������ �������������
test_image_path = r"C:\Users\myton\Downloads\archive (1)\fire_dataset\fire_images\fire.77.png"  # ������� ���� � ��������� �����������
predict_image(test_image_path)

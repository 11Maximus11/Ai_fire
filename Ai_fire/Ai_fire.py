# Импорт необходимых библиотек
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

# Укажите пути к папкам с изображениями
fire_path = r"C:\Users\myton\Downloads\archive (1)\fire_dataset\fire_images"       # Папка с изображениями пожара
non_fire_path = r"C:\Users\myton\Downloads\archive (1)\fire_dataset\non_fire_images"  # Папка с изображениями без пожара

# Проверяем, есть ли сохранённая модель
model_path = "fire_detection_model.h5"

if not os.path.exists(model_path):
    print("Модель не найдена, начнём обучение...")

    # Создание DataFrame
    data = []  # Временный список для хранения данных

    # Сбор данных о пожарах
    for dirname, _, filenames in os.walk(fire_path):
        for filename in filenames:
            data.append({'path': os.path.join(dirname, filename), 'label': 'fire'})

    # Сбор данных о случаях без пожара
    for dirname, _, filenames in os.walk(non_fire_path):
        for filename in filenames:
            data.append({'path': os.path.join(dirname, filename), 'label': 'non_fire'})

    # Преобразование списка в DataFrame
    df = pd.DataFrame(data)

    # Перемешиваем данные
    df = df.sample(frac=1).reset_index(drop=True)

    # Анализ данных
    sns.countplot(data=df, x='label')
    plt.title("Количество изображений по классам")
    plt.show()

    # Создание генераторов данных
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

    # Печать меток классов
    class_indices = train_gen.class_indices
    print(f"Метки классов: {class_indices}")  # {'fire': 0, 'non_fire': 1}

    # Создание модели
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

    # Компиляция модели
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Обучение модели
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=15,
        callbacks=[early_stopping, reduce_lr]
    )

    # Сохранение модели
    model.save(model_path)
    print("Модель сохранена!")
else:
    # Загрузка сохранённой модели
    model = load_model(model_path)
    print("Модель загружена!")

# Тестирование модели на новых изображениях
def predict_image(image_path):
    # Загрузка изображения
    img = load_img(image_path, target_size=(256, 256))  # Обновленный импорт
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Предсказание
    prediction = model.predict(img_array)
    label = 'fire' if prediction[0][0] < 0.5 else 'non_fire'
    print(f"Изображение {image_path}: {label}")

# Пример использования
test_image_path = r"C:\Users\myton\Downloads\archive (1)\fire_dataset\fire_images\fire.77.png"  # Укажите путь к тестовому изображению
predict_image(test_image_path)

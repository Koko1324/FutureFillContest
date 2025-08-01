import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# ===================== 설정 =====================
IMG_HEIGHT = 128  # 이미지 높이
IMG_WIDTH = 128   # 이미지 너비
BATCH_SIZE = 32
EPOCHS = 10

# ===================== 경로 설정 =====================
base_dir = 'dataset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# ===================== 데이터 전처리 =====================
# 학습용 제너레이터 (데이터 증강 포함)
train_datagen = ImageDataGenerator(
    rescale=1./255,              # 픽셀 값을 0~1로 정규화
    rotation_range=20,           # 20도 이내 회전
    zoom_range=0.2,              # 확대/축소
    horizontal_flip=True         # 수평 뒤집기
)

# 테스트용 제너레이터 (정규화만 적용)
test_datagen = ImageDataGenerator(rescale=1./255)

# 학습 데이터 불러오기
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'  # 이진 분류
)

# 테스트 데이터 불러오기
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# ===================== CNN 모델 구성 =====================
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # sigmoid: 이진 분류
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ===================== 모델 학습 =====================
model.fit(train_generator, epochs=EPOCHS, validation_data=test_generator)

# ===================== 모델 저장 =====================
model.save('rain_classifier_model.h5')  # 모델을 .h5 파일로 저장

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# 모델이 학습된 입력 이미지 크기
IMG_HEIGHT = 128
IMG_WIDTH = 128

# 모델 불러오기
model = load_model('rain_classifier_model.h5')

# 이미지 불러오기 및 전처리
img_path = 'imageSave/1.jpeg'  # 예측할 이미지 경로
img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))  # 크기 맞추기
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # 정규화
img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가

# 예측
prediction = model.predict(img_array)[0][0]  # 예측 확률
label = 'Rain' if prediction > 0.5 else 'No Rain'
print(f'예측 결과: {label} ({prediction:.2f})')

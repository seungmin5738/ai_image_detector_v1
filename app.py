# app.py (버그 수정된 최종 코드)

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# --- 1. AI 모델 로드 (한 번만 로드) ---
@st.cache_resource
def load_my_model():
    print("Loading model...")
    # Conda 환경의 tensorflow가 모델을 로드합니다.
    model = tf.keras.models.load_model('model_v1.h5')
    print("Model loaded.")
    return model

model = load_my_model()
class_names = ['FAKE', 'REAL'] # 0=FAKE, 1=REAL

# --- 2. 이미지 1장을 예측하는 함수 (버그 수정) ---
def predict_single_image(image_pil):
    """
    Streamlit에서 업로드한 PIL 이미지를 받아서 'FAKE' 또는 'REAL' 반환
    """
    # 1. PIL 이미지를 OpenCV(Numpy) 형식으로 변환 (RGB)
    img = np.array(image_pil)
    
    # 2. 32x32 크기로 리사이즈
    img_resized = cv2.resize(img, (32, 32))
    
    # 3. (버그 수정!) "이중 정규화" 버그를 위해 수동 정규화( / 255.0)를 삭제합니다.
    #    모델(model_v1.h5)이 "0~255" 범위의 원본 이미지를 기대합니다.
    #    img_normalized = img_resized / 255.0  <--- 이 줄을 삭제!

    # 4. 배치 차원 추가 (모델은 (1, 32, 32, 3) 형태를 기대)
    #    img_resized (0~255 범위)를 바로 넣습니다.
    img_batch = np.expand_dims(img_resized, axis=0)

    # 5. 예측
    prediction = model.predict(img_batch)
    score = prediction[0][0]
    
    if score < 0.5:
        return 'FAKE', score
    else:
        return 'REAL', score

# --- 3. 웹사이트 '얼굴' (Frontend) ---

st.title("🤖 Real or Fake? AI 이미지 탐지기")
st.write("32x32 픽셀의 TinyCNN으로 학습된 모델입니다.")
st.write("CIFAR-10(REAL) vs. Stable Diffusion(FAKE)")

uploaded_file = st.file_uploader("이미지 파일을 업로드하세요 (jpg, png)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. 사용자가 업로드한 이미지를 화면에 표시
    image = Image.open(uploaded_file)
    st.image(image, caption='업로드된 이미지', use_column_width=True)
    
    # 2. 예측 버튼
    if st.button("AI로 분석하기"):
        # 3. 로딩 스피너 표시
        with st.spinner('AI가 이미지를 분석 중입니다...'):
            # 4. AI 예측 실행
            label, score = predict_single_image(image)
        
        # 5. 결과 표시
        st.subheader("분석 결과:")
        if label == 'FAKE':
            st.error(f"이 이미지는 'FAKE' (AI 생성)일 확률이 높습니다.")
            # FAKE일 확률 (score가 0에 가까울수록 FAKE)
            st.write(f"(신뢰도: {(1-score)*100:.2f}%)")
        else:
            st.success(f"이 이미지는 'REAL' (실제 사진)일 확률이 높습니다.")
            # REAL일 확률 (score가 1에 가까울수록 REAL)
            st.write(f"(신뢰도: {score*100:.2f}%)")
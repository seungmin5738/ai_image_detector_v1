# app.py (오류 처리 + 신뢰도 3분할 최종 코드)

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# --- 1. AI 모델 로드 ---
@st.cache_resource
def load_my_model():
    print("Loading model...")
    model = tf.keras.models.load_model('model_v1.h5')
    print("Model loaded.")
    return model

model = load_my_model()

# --- 2. 이미지 1장을 예측하는 함수 (오류 처리 기능 강화) ---
def predict_single_image(image_pil):
    """
    PIL 이미지를 받아 score (0.0 ~ 1.0) 또는 None (오류)을 반환
    """
    try:
        # (핵심 수정 1: 모든 이미지를 RGB(3채널)로 강제 변환)
        # PNG(RGBA)나 흑백(L) 이미지가 들어와도 처리할 수 있게 함
        image_rgb = image_pil.convert('RGB')
        
        # PIL 이미지를 OpenCV(Numpy) 형식으로 변환
        img = np.array(image_rgb)
        
        # 32x32 크기로 리사이즈
        img_resized = cv2.resize(img, (32, 32))
        
        # (이중 정규화 버그가 수정된 상태)
        # 0~255 원본 픽셀을 배치로 만듦
        img_batch = np.expand_dims(img_resized, axis=0)

        # (핵심 수정 2: 최종 입력 형태 확인)
        # 모델이 (1, 32, 32, 3) 입력을 받았는지 확인
        if img_batch.shape != (1, 32, 32, 3):
            print(f"Image shape mismatch: {img_batch.shape}")
            return None # 오류 반환

        # 예측 실행
        prediction = model.predict(img_batch)
        score = prediction[0][0] # 0.0 ~ 1.0 사이 (REAL 확률)
        return score

    except Exception as e:
        # 이미지 처리 중 (예: 손상된 파일) 알 수 없는 오류가 발생하면...
        st.error(f"이미지 처리 중 오류 발생: {e}")
        return None # 오류 반환

# --- 3. 웹사이트 '얼굴' (Frontend) ---
st.title("🤖 Real or Fake? AI 이미지 탐지기")
st.write("32x32 픽셀의 TinyCNN으로 학습된 모델입니다.")
st.write("CIFAR-10(REAL) vs. Stable Diffusion(FAKE)")

uploaded_file = st.file_uploader("이미지 파일을 업로드하세요 (jpg, png)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='업로드된 이미지', use_column_width=True)
    
    if st.button("AI로 분석하기"):
        with st.spinner('AI가 이미지를 분석 중입니다...'):
            # score는 0.0 ~ 1.0 사이의 점수, 또는 None
            score = predict_single_image(image)
        
        st.subheader("분석 결과:")

        # (핵심 수정 3: 결과를 4가지 경우로 나눔)

        # 1. 예측 실패 (None)의 경우
        if score is None:
            st.error("이미지 분석에 실패했습니다.")
            # (사용자 요청 메시지)
            st.info("파일이 손상되었거나, 지원되지 않는 형식일 수 있습니다.\n\n" +
                     "다른 표준 JPG 또는 PNG 파일로 시도해주세요.")

        # 2. 확실한 REAL (예: 60% 이상)
        elif score > 0.6: 
            st.success(f"이 이미지는 'REAL' (실제 사진)일 확률이 높습니다.")
            st.write(f"(신뢰도: {score*100:.2f}%)")
            
        # 3. 확실한 FAKE (예: 40% 이하)
        elif score < 0.4:
            st.error(f"이 이미지는 'FAKE' (AI 생성)일 확률이 높습니다.")
            st.write(f"(신뢰도: {(1-score)*100:.2f}%)")
            
        # 4. 불확실 (40% ~ 60% 사이)
        else:
            st.warning(f"AI가 이미지를 판별하기 어렵습니다. (신뢰도 낮음)")
            st.write(f"({score*100:.2f}% REAL, {(1-score)*100:.2f}% FAKE)")
            # (사용자 요청 메시지)
            st.info("사진을 다른 각도로 찍어주세요.\n\n" +
                     "또는, 다른 추가적인 사진을 첨부해주세요.")
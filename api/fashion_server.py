# FastAPI: 웹 프레임워크
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# PyTorch 관련 모듈
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# 이미지 처리용
from PIL import Image
import io 

# 단계 1. 모델 정의 CNN
class ConvNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvNeuralNetwork, self).__init__()

        # 입력 feature map을 1차원으로 펼치는 층
        self.flatten = nn.Flatten()

        # 합성곱 신경망 구성 (CNN 구조 정의)
        self.classifier = nn.Sequential(
            nn.Conv2d(1, 28, kernel_size=3, padding='same'), 
            # 입력 채널 1(흑백), 출력 채널 28 (피쳐맵 28개), 커널 크기 3x3, 패딩은 출력 크기 유지(28x28)
            # 출력 크기: (28, 28, 28)
            nn.ReLU(),
            # 활성화 함수: max(0, x)로 음수 제거 → 비선형성 추가

            nn.Conv2d(28, 28, kernel_size=3, padding='same'),
            # 같은 깊이로 반복하여 추출된 특징을 심화
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            # 풀링 크기 2x2로 해당 영역에서 최대값만 추출
            # 특징을 강조하고 해상도를 절반으로 
            # 출력 크기: (28, 14, 14)
            nn.Dropout(0.25),
            # 학습 시 랜덤으로 25% 뉴런 비활성화하여 과적합을 방지한다.

            nn.Conv2d(28, 56, kernel_size=3, padding='same'),
            # 입력 28, 출력 56 (출력 채널을 증가시킴)
            # 더 많은 특징을 추출하여 표현력을 증가시킴
            # 출력 크기: (56, 14, 14)
            nn.ReLU(),

            nn.Conv2d(56, 56, kernel_size=3, padding='same'), 
            # 채널 56으로 유지. 깊은 특징을 학습
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            # 다시 절반 다운샘플링함. (56, 14, 14) -> (56, 7, 7)
            nn.Dropout(0.25), # 과적합 방지를 위한 드롭아웃
        ) 
        # 최종 feature map shape: (56, 7, 7)
        # Flatten 이후 → 56×7×7 = 2744차원 벡터
        self.Linear = nn.Linear(56 * 7 * 7, 10) # 선형 분류기: 최종 결과로 10개가 나온다. (10개 클래스)

    # 모델 순전파 정의
    def forward(self, x):
        x = self.classifier(x)  # CNN 통과
        x = self.flatten(x)     # 1차원으로 펼침
        output = self.Linear(x) # 분류 결과 출력 (logits)
        return output

# 단계 2. 모델 로딩 및 준비
model = ConvNeuralNetwork()

# 저장된 학습된 가중치 로드 (CPU 기준)
state_dict = torch.load('./model_weights.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval() # 추론 모드 (Dropout/BatchNorm 비활성화)

# 단계 3. 클래스 레이블 정의 (총 10개)
CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# 단계 4. 이미지 전처리 함수 정의
def preprocess_image(image_bytes):
     # torchvision 변환 정의: 리사이즈 → 텐서 변환 → 정규화
    transform = transforms.Compose([
        transforms.Resize((28, 28)),          # 모델 입력 크기에 맞게 리사이즈
        transforms.ToTensor(),                # 0~255 → 0~1 범위의 텐서로 변환
        transforms.Normalize((0.5,), (0.5,))  # 평균 0.5, 표준편차 0.5로 정규화 
    ])

    # 이미지 열기 (흑백으로 변환)
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    # 변환 적용 후 batch 차원 추가 (1, 1, 28, 28)
    return transform(image).unsqueeze(0)

# FastAPI 앱 생성
app = FastAPI()
# CORS 설정 (모든 도메인에서 요청 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 모든 origin 허용
    allow_credentials=True,
    allow_methods=["*"],   # 모든 HTTP 메서드 허용
    allow_headers=["*"],   # 모든 헤더 허용
)

# 이미지 분류 API 라우트 정의
@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read() # 파일 바이트 읽기
        print(f"Received file: {file.filename}, size: {len(image_bytes)} bytes")
        
        input_tensor = preprocess_image(image_bytes) # 이미지 전처리 → 텐서로 변환
        print(f"input tensor shape: {input_tensor.shape}") # e.g., torch.Size([1, 1, 28, 28])

        # 추론 (autograd 끔)
        with torch.no_grad():
            outputs = model(input_tensor)
            print(f"Model outputs: {outputs}") # 모델 출력 logits

            # 가장 높은 확률을 갖는 클래스 선택
            _, predicted = torch.max(outputs, 1)
            label = CLASSES[predicted.item()]
            print(f"Predicted label: {label}")
            
        # 결과 반환
        return JSONResponse(content={"label": label})
        
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

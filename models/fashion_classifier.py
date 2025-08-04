import torch                # PyTorch의 핵심 모듈 (텐서, GPU 연산 등 제공)
import torch.nn as nn       # PyTorch의 신경망 모듈 (모델 구성, 레이어 정의 등)
import torch.optim as optim # 최적화 알고리즘 모듈 (예: SGD, Adam 등)
from torch.utils.data import DataLoader       # 데이터 배치 처리 및 로딩 도구 (DataLoader, Dataset 등)
import torchvision.transforms as transforms   # 이미지 전처리 및 변환 도구 (Resize, ToTensor, Normalize 등)
from torchvision.datasets import FashionMNIST # FashionMNIST 제공


# 훈련용 이미지 전처리
train_transform = transforms.Compose([
    transforms.Resize((28, 28)),    # 이미지 크기를 28x28로 조정 (FashionMNIST 기본 크기와 맞춤)
    transforms.Grayscale(1),        # 채널 수를 1로 강제 변환 (흑백 이미지로 처리, 채널=1)
    transforms.RandomInvert(p=0.5), # 50% 확률로 픽셀 값을 반전 (0↔1), 데이터 다양성 증가 (data augmentation)
    transforms.ToTensor(),          # 이미지를 PyTorch 텐서로 변환 (픽셀값을 0~1 범위의 float32로 정규화)
    transforms.Normalize((0.5,), (0.5,)) # 평균 0.5, 표준편차 0.5로 정규화 → -1~1 범위로 분포 조정
])
# 테스트 이미지 전처리
test_transform = transforms.Compose([
    transforms.Resize((28, 28)),          # 테스트 이미지도 크기 동일하게 맞춤
    transforms.Grayscale(1),              # 흑백 이미지로 변환
    transforms.ToTensor(),                # 텐서로 변환
    transforms.Normalize((0.5,), (0.5,))  # 훈련과 동일한 정규화 적용 (훈련/테스트 일관성 유지)
])

# 데이터셋 로딩
data_path='C:\\Dana\\CNN\\data\\'

# 훈련 데이터셋
trainset = FashionMNIST(
    root=data_path,           # 데이터를 저장하거나 불러올 기본 경로 (예: 'C:\\Dana\\CNN\\data\\')
    train=True,               # 훈련용 데이터셋을 불러옴 (True: train set, False: test set)
    download=True,            # 데이터셋이 없으면 자동으로 다운로드
    transform=train_transform # 앞서 정의한 훈련 데이터 전처리(transform) 적용
)
    

# 테스트 데이터셋
testset = FashionMNIST(
    root=data_path,           # 같은 경로에 테스트 데이터도 저장
    train=False,              # 테스트용 데이터셋을 불러옴
    download=True,            # 데이터셋이 없으면 자동으로 다운로드
    transform=test_transform  # 앞서 정의한 테스트 전용 전처리(transform) 적용
    )


# 클래스 인덱스 → 이름 매핑
class_map = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# 데이터 로더 생성
loader = DataLoader(
    dataset = trainset, # 앞서 생성한 FashionMNIST 훈련 데이터셋 객체
    batch_size = 64,    # 한 번에 모델에 전달할 데이터 수 (배치 크기)
    shuffle=True        # 각 epoch마다 데이터 순서를 무작위로 섞어 학습 (모델의 일반화 성능 향상)
)

# 데이터 배치 확인
imgs, labels = next(iter(loader)) 
    # iter(loader)
        # loader는 PyTorch의 DataLoader 객체
        # DataLoader : 반복 가능한 객체(iterable)
        # iter()를 사용하면 이터레이터(iterator) 반환
    # next() : 이터레이터에서 다음 값을 꺼내는 함수
print(imgs.shape, labels.shape) # (배치 크기, 채널, 높이, 너비)

# CNN 모델 정의
class ConvNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvNeuralNetwork, self).__init__()

        # 완전연결층 입력을 위해 평탄화(flatten) 도구 준비
        self.flatten = nn.Flatten()

        # 합성곱 기반 분류기 (Sequential: 레이어들을 순서대로 쌓음)
        self.classifier = nn.Sequential(

            # [입력: 1채널, 출력: 28채널], 커널 3x3, 패딩 same → 출력 크기 유지
            nn.Conv2d(1, 28, kernel_size=3, padding='same'),
            nn.ReLU(), # 비선형 활성화 함수

            # [입력: 28채널, 출력: 28채널], 추가 합성곱
            nn.Conv2d(28, 28, kernel_size=3, padding='same'),
            nn.ReLU(),

            # [Downsampling] 2x2 영역에서 최대값만 선택 → 크기 절반 감소 (28x28 → 14x14)
            nn.MaxPool2d(kernel_size=2),

            # 과적합 방지를 위한 드롭아웃 (25% 확률로 무작위 노드 제거)
            nn.Dropout(0.25),

            # 채널 수 증가: 28 → 56 (특징 추출 강화)
            nn.Conv2d(28, 56, kernel_size=3, padding='same'),
            nn.ReLU(),

            nn.Conv2d(56, 56, kernel_size=3, padding='same'), 
            nn.ReLU(),

            # 또 한 번 다운샘플링 (14x14 → 7x7)
            nn.MaxPool2d(kernel_size=2), 
            nn.Dropout(0.25),
        )
        # 최종 완전연결층 (Flatten 후 56채널 × 7 × 7 크기의 벡터 → 클래스 10개로 분류)
        self.Linear = nn.Linear(56 * 7 * 7, 10) # 10개 클래스 분류
    
    def forward(self, x):
        # 입력 이미지 x → CNN 연산 → Flatten → FC → 출력 벡터 반환
        x = self.classifier(x)  # CNN + 활성화 + 풀링 + 드롭아웃
        x = self.flatten(x)     # 2D feature map을 1D 벡터로 변환
        output = self.Linear(x) # 분류 결과 (로짓, softmax는 나중에 따로)
        return output


# 학습 장치 설정 및 모델 준비
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model = ConvNeuralNetwork().to(device)
print(model)

# 손실 함수 및 옵티마이저 정의
loss = nn.CrossEntropyLoss()
# → 다중 클래스 분류에서 자주 사용하는 손실 함수
# → 모델 출력 (logits)과 정답 라벨 간의 차이를 계산함
# → 내부적으로 Softmax + NLLLoss 조합과 같음

optimizer = optim.Adam(model.parameters(), lr=0.001)
# → Adam: 적응형 학습률을 가진 효율적인 최적화 알고리즘
# → model.parameters(): 학습 대상 파라미터 (가중치, 편향 등)
# → lr: 학습률(learning rate). 값이 작을수록 천천히, 클수록 빠르게 학습됨

# 학습 루프 함수 정의
def train_loop(train_loader, model, loss_fn, optimizer):
    sum_losses = 0 # 전체 손실 합 (평균 손실 계산용)
    sum_accs = 0   # 전체 정확도 합 (평균 정확도 계산용)

    for x_batch, y_batch in train_loader: # DataLoader로 배치 단위 반복
        x_batch = x_batch.to(device)      # 입력 텐서를 GPU or CPU로 이동
        y_batch = y_batch.to(device)      # 정답 라벨도 동일한 장치로 이동

        y_pred = model(x_batch)           # 모델에 입력하여 예측값(logits) 계산
        loss = loss_fn(y_pred, y_batch)   # 예측값과 정답 간의 손실 계산

        optimizer.zero_grad()             # 이전 배치의 gradient 초기화
        loss.backward()                   # 손실에 대한 gradient 계산 (역전파)
        optimizer.step()                  # 계산된 gradient로 가중치 업데이트

        sum_losses = sum_losses + loss    # 배치 손실 누적

        # 예측 정확도 계산
        y_prob = nn.Softmax(1)(y_pred)    # 차원 1 기준 softmax → 클래스 확률 벡터로 변환
        y_pred_index = torch.argmax(y_prob, axis=1) # 확률이 가장 높은 인덱스를 예측값으로 선택
        acc = (y_batch == y_pred_index).float().sum() / len(y_batch) * 100
        # → 예측이 정답과 일치하는 비율을 백분율로 계산
        sum_accs = sum_accs + acc # 배치 정확도 누적
    
    # 전체 평균 손실과 정확도 계산 후 반환
    avg_loss = sum_losses / len(train_loader)
    avg_acc = sum_accs / len(train_loader)
    return avg_loss, avg_acc

# 학습 실행
epochs = 50 # 총 학습 반복 횟수 설정

for i in range(1, epochs+1, 10): # 10 epoch 단위로 출력
    print(f"------------------------------------------------")
    avg_loss, avg_acc = train_loop(loader, model, loss, optimizer)
    # train_loop(): 훈련 데이터 한 바퀴 돌면서 → 모델 학습 + 정확도 측정하는 함수
    print(f'Epoch {i:4d}/{epochs} Loss: {avg_loss:.6f} Accuracy: {avg_acc:.2f}%')
print("Done!")


# 테스트 데이터 평가
test_loader = DataLoader(
    dataset=testset, # FashionMNIST 테스트셋 사용
    batch_size=32,   # 작은 배치 단위로 평가
    shuffle=False    # 순서 유지 (정확도 평가만 하므로 섞을 필요 없음)
)

# 테스트 함수 정의
def test(model, loader):
    model.eval() # 평가 모드로 전환 (Dropout, BatchNorm 등 비활성화)

    sum_accs = 0 # 전체 정확도 누적용 변수

    # 모든 예측/정답/이미지를 저장할 텐서 초기화
    img_list = torch.Tensor().to(device)    # 입력 이미지 전체 저장용
    y_pred_list = torch.Tensor().to(device) # 예측된 클래스 인덱스 저장용
    y_true_list = torch.Tensor().to(device) # 실제 정답 인덱스 저장용

    for x_batch, y_batch in loader: 
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(x_batch)                     # 모델 예측 (로짓)
        y_prob = nn.Softmax(1)(y_pred)              # 확률값으로 변환
        y_pred_index = torch.argmax(y_prob, axis=1) # 가장 높은 확률의 클래스 선택

        # 결과 저장
        y_pred_list = torch.cat((y_pred_list, y_pred_index), dim=0)
        y_true_list = torch.cat((y_true_list, y_batch), dim=0)
        img_list = torch.cat((img_list, x_batch), dim=0)
        # torch.cat() : 여러 개의 텐서를 하나로 연결(concatenate) 해주는 함수
        # dim=0: 첫 번째 차원(batch 차원)을 기준으로 붙임
        # 이어붙이는 이유
            # test_loader는 데이터를 여러 배치로 나눠서 줌
            # 테스트 결과를 분석하거나 시각화할 땐 전체 결과가 한 텐서에 있어야

        # 현재 배치 정확도 계산
        acc = (y_batch == y_pred_index).float().sum() / len(y_batch) * 100
        sum_accs += acc
    
    # 전체 평균 정확도 계산
    avg_acc = sum_accs / len(loader)
    return y_pred_list, y_true_list, img_list, avg_acc

y_pred_list, y_true_list, img_list, avg_acc = test(model, test_loader)

print(f'테스트 정확도는 {avg_acc:.2f}% 입니다.')

# 모델 저장
torch.save(model.state_dict(), 'model_weights.pth') # 가중치만 저장
torch.save(model, 'model.pt')                       # 전체 모델 저장
## 생성적 딥러닝 모델 기반 설비 이상 탐지 프로그램

### 본 프로그램 특징
   - 플랜트 내 센서 데이터를 기반으로 설비 이상을 탐지하는 프로그램
   - 정상 데이터만을 가지고 이상 여부를 판단하는 비지도 학습 기반의 생성 모델 AnoGAN을 사용
   - AnoGAN은 정상 데이터를 학습하고 생성하는 'GAN(Generative Adversarial Network)'과 입력 받은 데이터와 잠재 벡터로부터 GAN이 생성한 데이터 샘플 간 차이를 계산하여 이상치 점수를 계산하는 'Loss Funtion'으로 구성
     
   - 본 프로그램은
     - (1) 사전에 학습된 모델을 불러와서 플랜트 내 태그 센서 1D 데이터를 입력하면, 생성 모델이 입력 데이터와 가장 유사한 데이터를 생성하여,
     - (2) 생성된 데이터의 Loss 값을 기반으로 이상 여부를 판단함
   - 프로그램 입력 : 설비 데이터 (256크기의 1차원의 numpy array)
   - 프로그램 출력 : 이상치 점수 
   - 프로그램 완료시, 평가 대상 데이터셋에 대한 이상치 점수 리스트가 .npy 형식으로 코드 파일 경로에 저장됨
        
### 주요 기능
   - 플랜트 내 센서 데이터를 입력받아 설비 이상을 탐지
   - 모델의 이상 점수 분포를 기반으로 정상 데이터와 이상 데이터를 구분
     
### 사용 방법 
   - 평가 대상 데이터 경로를 ANOGAN.py 에 test_path 인자 값으로 넣어 설비 이상을 탐지
   - AnoGAN 내 WGAN 모델 학습 필요시, training 인자를 True로 하여 학습 가능
     - 모델 새로 학습 시, 학습 데이터 형태 (1차원의 numpy arrays)
     - 학습 epoch 수, batch 크기 설정 가능
   - 입력 옵션 목록 (옵션 플래그 입력 형식)
     - training	(--training 'bool' )	    	: WGAN 모델 학습 실행 유무, default False  
     - EPOCH	(--epoch 'int') 		: 모델 학습 EPOCH 수, 정수형, default 500
     - BATCH_SIZE	(--batch_size 'int')		: 학습 batch 크기, 정수형, default 32
     - raw_path 	(--raw_path 'path' )		: raw 데이터(wav file) 경로, defalut 'raw/'
     - train_path 	(--train_path 'path')	: train 데이터와 test 데이터 분할 시 train 데이터 저장 경로 혹은 이미 저장된 train 데이터 경로, default 'train_data/'   
     - test_path	(--test_path 'path') 	: train 데이터와 test 데이터 분할 시 test 데이터 저장 경로 혹은 이미 저장된 test 데이터 경로, default 'test_data/'

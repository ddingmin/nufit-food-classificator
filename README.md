# nufit-food-classificator
torch의 ResNet50을 이용해 restAPI로 구현한 음식 분류 AI 모델 서빙 서버.

음식 이미지를 응답받아 Image Classification 인퍼러스 결과 상위 5개를 응답.
## 동작 환경
```
Python = 3.8.16
torch == 2.0.1
torchvision == 0.15.2
```

## 환경 세팅
```Bash
pip install -r requirements.txt
```

```/model``` 디레토리에 가중치 파일을 넣고 ```inference.py``` 파일의 path를 지정해서 사용하면 된다.
## 예시 사진
1. 이미지 인퍼런스 요청 (초콜릿 이미지)

![초콜릿](/images/chocolate.jpeg)

- 결과

![초콜릿 결과](/images/chocolate_result.png)

2. 학습된 클래스 리스트 요청

![클래스_리스트](/images/class_list.png)

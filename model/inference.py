import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

labels = ['가츠동', '갈비', '갈비탕', '감자구이', '감자그라탕', '감자칩', '감자탕', '감자튀김', '게맛살', '게장', '고구마', '고로케', '곰탕', '국수', '그라탕', '김밥',
          '김치국', '김치볶음밥', '김치찌개', '나시고랭', '낚지볶음', '낫또', '냉면', '단호박', '달걀', '닭가슴살구이', '닭갈비', '닭고기볶음', '닭찜', '닭칼국수', '도넛',
          '돈가스', '돼지고기구이', '두부', '딸기', '떡볶이', '라멘', '라면', '랍스타', '리조또', '마들렌', '마시멜로우', '마카롱', '막국수', '만두', '만두국', '망고',
          '머핀', '메쉬드포테이토', '멜론', '무김치', '물냉면', '미소장국', '미트볼', '밀감', '바게트빵', '바나나칩', '밤', '밥', '방울토마토', '배추김치', '베이글',
          '보쌈', '복숭아', '볶음면', '볶음밥', '부대찌개', '부침개', '불고기덮밥', '붕어빵', '블루베리', '비빔냉면', '비빔밥', '뻥튀기', '뼈해장국', '사과', '사과파이',
          '삼계탕', '상추', '새우', '새우튀김', '샌드위치', '샐러드', '생선구이', '생선튀김', '생선회', '석류', '소곱창구이', '소시지', '송편', '쇠고기구이', '수박',
          '수제비', '순대국', '순대볶음', '슈크림', '스프', '시리얼바', '쌀국수', '아이스크림', '아포가토', '알밥', '야끼소바', '야채볶음', '양꼬치', '양념치킨',
          '에그타르트', '오렌지', '오므라이스', '오징어덮밥', '오코노미야끼', '오트밀', '와플', '우동', '육개장', '육회', '자두', '자몽', '전복', '젤리', '족발',
          '주먹밥', '죽', '짜장면', '짬뽕', '쨈빵', '쪽갈비구이', '쭈꾸미볶음', '찐빵', '찜닭', '참치통조림', '청포도', '체리', '초밥', '초코머핀', '초코쿠키',
          '초콜릿', '츄러스', '치즈볼', '치킨', '치킨너겟', '카레라이스', '캘리포니아롤', '컵라면', '케이크', '쿠키', '크래커', '크레페', '크로와상', '키위', '타코야키',
          '탕수육', '토마토', '토스트', '통밀빵', '티라미수', '파스타', '파인애플', '파전', '팝콘', '팟타이', '포도', '폭립', '푸딩', '프레즐', '프렌치토스트', '피자',
          '함박스테이크', '핫도그', '핫윙', '핫케이크', '해시브라운', '호두파이', '후라이드치킨', '훈제오리']

# init - resnet50
model = models.resnet101(pretrained=False)
num_classes = len(labels)
weight_path = './model/ResNet101.pth'

model.fc = torch.nn.Linear(2048, num_classes)
checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'])
model.eval()

# init - transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
])


def inference(input_image):
    input_image = Image.open(input_image).convert('RGB')
    input_data = transform(input_image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_data)

    predicted_classes = torch.sigmoid(output)
    top_5_values, top_5_indices = torch.topk(predicted_classes, k=5)

    result = []
    for i in top_5_indices[0]:
        result.append(labels[i])

    return result


def get_classes():
    return {"class": labels}

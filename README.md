# level2-data-annotation_nlp-level2-nlp-10

## ❗ 주제 설명
- '컴퓨터언어' 주제로 데이터셋 만들기
- KLUE를 참고하여 총 10개의 relation으로 구분된 데이터셋 작성

## 👋 팀원 소개
### Members
|김남현|민원식|전태양|정기원|주정호|최지민|
|:-:|:-:|:-:|:-:|:-:|:-:|
|<img src='https://user-images.githubusercontent.com/73579424/164642575-4273ba4f-f291-4f44-b37b-856ecb8df450.png' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/73579424/164642795-b5413071-8b14-458d-8d57-a2e32e72f7f9.png' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/73579424/164642916-2ba2c870-9773-44c3-9acd-b3ac46d77d2a.png' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/73579424/164643061-599b9409-dc21-4f7a-8c72-b5d5dbfe9fab.jpg' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/73579424/164643280-b0981ca3-528a-4c68-9331-b8f7a1cbe414.jpg' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/73579424/164643383-cf02b20e-07b7-4f50-bb79-e3d1cf5db084.png' height=80 width=80px></img>|
|[Github](https://github.com/NHRWV)|[Github](https://github.com/wertat)|[Github](https://github.com/JEONSUN)|[Github](https://github.com/greenare)|[Github](https://github.com/jujeongho0)|[Github](https://github.com/timmyeos)|

### Members' Role
| 팀원 | 역할 | 
| --- | --- |
| 김남현(T3021) | Relation map/가이드라인 작성 & 어노테이션 작업 |
| 민원식(T3079) | Relation map/가이드라인 작성 & 어노테이션 작업 |
| 정기원(T3195) | Relation map/가이드라인 작성 & 어노테이션 작업 |
| 주정호(T3211) | Relation map/가이드라인 작성 & 어노테이션 작업 & Fleiss’ Kappa 측정 & Fine-Tunning  |
| 최지민(T3223) | Relation map/가이드라인 작성 & 어노테이션 작업 |

## 🔨 Installation

아래 사항들로 현 프로젝트에 관한 모듈들을 설치할 수 있습니다.

```
pandas==1.1.5
scikit-learn~=0.24.1
transformers==4.10.0
requests
html
zipfile
shutil
json
```

## ✍ Function Description

`train.py`: "klue/roberta-large"을 바탕으로 LSTM layer를 추가하여 model을 생성하고 주어진 train dataset을 통해 train 진행

`inference.py`: "klue/roberta-large"을 바탕으로 LSTM layer를 추가하여 model을 생성하고 주어진 test dataset을 통해 inference 진행

`load_data.py + 'df_edit.py'`: 주어진 dataset에서 원하는 항목을 분리하고 type-entity 등을 추가

`tagtog2csv.py`: tagtog에서 진행한 annotation 작업물을 request를 통해 받아와서 KLUE 데이터셋의 양식으로 편집하여 저장

`calculate_iaa.py + fleiss.py`: Fleiss Kappa를 계산


## 🏢 Structure

```bash
level1-image-classification-level1-recsys-09
│
├── README.md
├── requirements.txt
├── EDA
│   ├── data_EDA.ipynb
│   ├── image_EDA.ipynb
│   └── torchvision_transforms.ipynb
└── python
    ├── train.py
    ├── inference.py
    ├── load_data.py
    ├── df_edit.py
    ├── tagtog2csv.py
    ├── calculate_iaa.py
    └── fleiss.py
```

## Deployment / 배포

Add additional notes about how to deploy this on a live system / 라이브 시스템을 배포하는 방법

## Built With / 누구랑 만들었나요?

* [이름](링크) - 무엇 무엇을 했어요
* [Name](Link) - Create README.md

## Contributiong / 기여

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us. / [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) 를 읽고 이에 맞추어 pull request 를 해주세요.

## License / 라이센스

This project is licensed under the MIT License - see the [LICENSE.md](https://gist.github.com/PurpleBooth/LICENSE.md) file for details / 이 프로젝트는 MIT 라이센스로 라이센스가 부여되어 있습니다. 자세한 내용은 LICENSE.md 파일을 참고하세요.

## Acknowledgments / 감사의 말

* Hat tip to anyone whose code was used / 코드를 사용한 모든 사용자들에게 팁
* Inspiration / 영감
* etc / 기타

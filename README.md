# level2-data-annotation_nlp-level2-nlp-10

## ❗ 주제 설명
- **'컴퓨터언어'** 를 주제로 RE(Relation Entity) 데이터셋 제작
- KLUE를 참고하여 총 10개의 relation으로 구분된 데이터셋 제작
- 직접 제작한 RE 데이터셋을 모델에 적용해보고 성능 검증

## 👋 팀원 소개
### Members
|김남현|민원식|전태양|정기원|주정호|최지민|
|:-:|:-:|:-:|:-:|:-:|:-:|
|<img src='https://avatars.githubusercontent.com/u/54979241?v=4' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/73579424/164642795-b5413071-8b14-458d-8d57-a2e32e72f7f9.png' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/55140109?v=4' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/73579424/164643061-599b9409-dc21-4f7a-8c72-b5d5dbfe9fab.jpg' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/73579424/164643280-b0981ca3-528a-4c68-9331-b8f7a1cbe414.jpg' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/97524127?v=4' height=80 width=80px></img>|
|[Github](https://github.com/NHRWV)|[Github](https://github.com/wertat)|[Github](https://github.com/JEONSUN)|[Github](https://github.com/greenare)|[Github](https://github.com/jujeongho0)|[Github](https://github.com/timmyeos)|

### Members' Role
| 팀원 | 역할 | 
| --- | --- |
| 김남현(T3021) | Relation map / 가이드라인 작성 & 어노테이션 작업 |
| 민원식(T3079) | Relation map / 가이드라인 작성 & 어노테이션 작업 |
| 전태양(T3194) | Relation map / 가이드라인 작성 & 어노테이션 작업 |
| 정기원(T3195) | Relation map / 가이드라인 작성 & 어노테이션 작업 |
| 주정호(T3211) | Relation map / 어노테이션 작업 & Fleiss’ Kappa 측정 & Fine-Tunning  |
| 최지민(T3223) | Relation map / 가이드라인 작성 & 어노테이션 작업 & 제작된 데이터 EDA |

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
`EDA+Data Viz.ipynb`: 데이터 EDA 및 시각화

`train.py`: "klue/roberta-large"을 바탕으로 LSTM layer를 추가하여 model을 생성하고 주어진 train dataset을 통해 train 진행

`inference.py`: "klue/roberta-large"을 바탕으로 LSTM layer를 추가하여 model을 생성하고 주어진 test dataset을 통해 inference 진행

`load_data.py + 'df_edit.py'`: 주어진 dataset에서 원하는 항목을 분리하고 type-entity 등을 추가

`tagtog2csv.py`: tagtog에서 진행한 annotation 작업물을 request를 통해 받아와서 KLUE 데이터셋의 양식으로 편집하여 저장

`calculate_iaa.py + fleiss.py`: Fleiss Kappa를 계산


## 🏢 Structure

```bash
level2-data-annotation_nlp-level2-nlp-10
│
├── README.md
├── requirements.txt
├── iaa.csv
├── train.csv
├── test.csv
│
│
├── EDA+Data Viz
│   └── EDA+Data Viz.ipynb
│   
│   
└── python
    ├── train.py
    ├── inference.py
    ├── load_data.py
    ├── df_edit.py
    ├── tagtog2csv.py
    ├── calculate_iaa.py
    └── fleiss.py
```

## 📂 Relation

|id	|class_name (ko)	|class_name (en)	|direction (sub, obj)	|description|
|---|---|---|---|---|
|1	|관계_없음	|no_relation	|(*, *)	|관계를 유추할 수 없음. 정의된 클래스 중 하나로 분류할 수 없음|
|2	|인물:제작	|per:production	|(PER, POH / COM)	|Object는 Subject가 제작한 것|
|3	|인물:직업/직함	|per:title	|(PER, POH)	|Object는 Subject의 직업/직함|
|4	|단체:제작	|org:production	|(ORG, POH / COM)	|Object는 Subject가 제작한 것|
|5	|기술:제작_날짜	|com:date_of_produced	|(COM, DAT)	|Object는 Subject가 제작된 날짜|
|6	|기술:하위_개념	|com:sub_concept	|(COM, COM / POH)	|Object는 Subject의 하위 개념|
|7	|기술:별칭	|com:alternative_names	|(COM, COM / POH)	|Object는 Subject의 또다른 이름|
|8	|기술:도구	|com:made_of	|(COM, COM)	|Object는 Subject를 만든(e.g. 작성, 개발, 구현한) 기술|
|9	|기술:선행_기술	|com:prior_technology	|(COM, COM)	|Object는 명시적으로(e.g. 근간을 두다, 기반하다.) Subject보다 앞선 기술|
|10	|기술:유사_기술	|com:similar_technology	|(COM, COM)	|Object는 명시적으로 Subject와 어떠한 공통 성질을 보유한 기술|



## 📋 Report
- [NLP] 데이터 제작 대회 WrapUP 리포트(노션) : [데이터 제작 대회_NLP_팀 리포트(10조).pdf](https://catnip-pelican-5b8.notion.site/_NLP_-10-9e4a94b82c114f7496ff429d79eafa21)
- [NLP] 데이터 제작 대회 WrapUP 리포트(PDF 파일 다운로드) : [NLP_데이터제작_Wrap_up_Report(10조).pdf](https://github.com/boostcampaitech3/level2-data-annotation_nlp-level2-nlp-10/files/8541329/NLP_._Wrap_up_Report.10.pdf)
- 최종제출물(가이드라인, relation map, 데이터셋) : [10조(핫식스)-컴퓨터언어-20220422T080136Z-001.zip](https://github.com/boostcampaitech3/level2-data-annotation_nlp-level2-nlp-10/files/8541129/10.-.-20220422T080136Z-001.zip)


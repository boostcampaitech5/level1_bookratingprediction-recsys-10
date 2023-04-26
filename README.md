# lv1 : RecSys-10 '대가족'

<br>

**프로젝트 개요**  

```
사용자의 책 평점 데이터를 바탕으로 사용자가 어떤 책을 더 선호할지 평점을 예측해 본다.
```
<br>

**프로젝트 수행방법**

1. 데이터 탐색(EDA) 및 전처리
2. 모델 탐구
3. 모델 선정
4. 모델 평가 및 개선
5. 결과 정리 및 분석

<br>

**데이터**

- 306,795건의 평점 데이터(train_rating.csv)
- 149,570건의 책 정보(books.csv)
- 68,092명의 고객 정보(users.csv)


<br>

## 데이터 탐색(EDA) 및 전처리

팀원들마다 각자 진행한 전처리 과정에서 최종적으로 선택한 방식이다.

1. text preprocessing

    - 불필요한 문장부호 제거, 모두 소문자화 등

2. column 별 전처리

    - 빈도수 낮은 범주들 병합
    <br>

    | column | 변형 | 결측치 |
    | :-------: | :-----------: | :-----------: |
    | Location | city, state, country 분할 <br> country 추출(usa는 sate 추출) | 같은 city는 같은 country로 통일 <br> 최빈값으로 채우기 |
    | Age | 범주형으로 맵핑 | 평균값과 catboost로 채우기 |
    | Language | 최빈값으로 채우기 |  |
    | Category |  | 최빈값으로 채우기 |
    | Book Author |  | 'Anonymous'로 채우기 |
    | Summary |  | 'None'으로 채우기 |


<br>

## 모델

1. 탐구 모델
    - FM, FFM
    - NCF, WDN, DCN
    - CNN_FM
    - DeepCoNN
    - CatBoost

<br>

2. 모델 선정
  ```
  CNN_FM, CatBoost
  ```

3. 모델 및 결과 개선

    - Grid-Search를 이용하여 Hyperparameter Tuning
    - 결과 앙상블


<br>

## 최종 제출

CatBoost 4개 앙상블 + CNN_FM 7개 앙상블 + CatBoost

<br>

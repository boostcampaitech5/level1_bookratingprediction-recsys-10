# (진유) EDA 및 전처리

# 1. Users

![image](https://user-images.githubusercontent.com/77721219/233972662-1712c7b4-b2f9-4dc3-accb-055272504fce.png)

위 users dataset의 information을 통해 알 수 있듯이 변수 age의 결측값이 많다.

이를 해결하기 전 다양한 EDA를 시도했다. 

### 가설 1. age가 nan 인 유저들 사이에 공통점이 있을 것이다.

(1) 동화책과 같은 어린이 책들은 아이들이 직접 구매하지 못하므로, 구매자가 age 정보를 선택하지 않았을 것이라 생각하여, ****************age가 nan인 유저들이 구매한 책의 표지와 category를 확인하였다.****************

![image](https://user-images.githubusercontent.com/77721219/233973202-56d0059d-7791-4cd9-84f5-c98d0908ecfd.png)

표지만으로는 확인하기 어려웠다.

category의 경우, category 종류를 확인하면서, ‘child’가 포함된 category가 있는 것을 확인하였다. 

 ‘child’가 포함된 category가 많이 있는지 확인하기 위해 age가 nan인 유저들이 구매한 책들 중 가장 많이 구매한 category 상위 10개를 출력해봤다.

![image](https://user-images.githubusercontent.com/77721219/233973360-17a8a406-dbf4-4a84-9e75-546fede9592e.png)

‘child’가 포함된 category는 상위 10개에 들지 못했지만 ‘Juvenile’이 포함된 category가 많이 발견되었다. (Juvenile은 5-18세를 의미한다.)

‘Fiction’ 은 아래 plot과 같이 압도적으로 많기 때문에 age 변수의 영향을 크게 받지 않을 것이라 생각했다.

![image](https://user-images.githubusercontent.com/77721219/233973462-8d9c360f-9f6c-4363-9ce5-9fed433aa718.png)

Juvenile 서적에 평점을 남긴 유저들의 나이 결측치를 처리하기 전에, 현재 users 데이터셋의 age 분포를 확인하였다.

### **가설 2. 5~18세는 직접 온라인으로 책을 구매하지 않을 것이다. 즉, rating은 책을 읽은 5~18세인 사람이 매기더라도, user 정보의 age는 책을 보호자가 대신 구매하였으므로 nan으로 처리되지 않았을까?**

age 분포를 확인해보면 10대가 두번째로 작다.

![image](https://user-images.githubusercontent.com/77721219/233973540-e8aa1f76-f43f-4a23-9374-7f80821cb2b9.png)

하지만 상식적으로 생각했을 때, 10대와 20대가 책을 읽는 경우가 많지 않나하는 생각이 들었다. 초~고등학교만 생각해도 학교에서 학생들이 책을 읽도록 권장하고, 부모님들이 아이들의 교육을 위해 많은 책을 읽히는 경우가 많다.

이 가설에 대한 근거자료를 찾기 위해, 전세계 연령별 독서량을 찾아보았다.

그 전에 범위를 전세계로 둘 경우 정확도가 떨어질 것이라 생각하여, age가 nan인 유저들의 국가 분포를 확인해봤다. 아래 plot을 통해 usa가 압도적으로 많은 것을 확인할 수 있다.

![image](https://user-images.githubusercontent.com/77721219/233973646-45426eaf-4a2d-4778-b5f8-01ace67014a7.png)

usa의 연령별 독서량을 찾아봤다.

![image](https://user-images.githubusercontent.com/77721219/233973777-cdc6e5ca-91bc-4a57-82d4-f7b228d32acd.png)

(출처: [https://www.forbes.com/sites/daviddisalvo/2012/10/23/who-reads-the-most-in-the-us-the-answer-might-surprise-you/?sh=5efa86e928b2](https://www.forbes.com/sites/daviddisalvo/2012/10/23/who-reads-the-most-in-the-us-the-answer-might-surprise-you/?sh=5efa86e928b2))

제일 마지막 문단에서 알 수 있듯이, 18-24세의 독서량이 가장 많고, 그 다음이 16-17세, 다음이 30-39세 인 것으로 나타났다.

위의 정보대로라면, 10대의 count가 20, 30대만큼 많아야하지만 그렇지 않다.

우리가 갖고 있는 users, books, ratings 데이터 즉, 유저들이 책을 구매한 이력은 usa 독서량을 온전히 표현한다고 할 수 없다.

하지만 age가 nan인 유저들이 구매한 서적들 중, Juvenile 서적을 구매한 유저는 약 2000명 정도밖에 되지 않으므로 모두 age를 juvenile로 처리하기로 했다. (자세한 전처리는 다음 전처리 페이지 참고)

바로 아래 plot과 같이 10대 유저가 늘어났다.

![image](https://user-images.githubusercontent.com/77721219/233974391-7f3c58e2-3c81-4fb4-904b-fbf8150ce8ef.png)


처리 전:

![image](https://user-images.githubusercontent.com/77721219/233974495-c9ee002e-77c7-435c-9252-c32b8f620979.png)

처리 후:

![image](https://user-images.githubusercontent.com/77721219/233974560-1aae7c18-e653-4974-a700-5171d080303c.png)

### 가설 3. location, age 모두 nan인 사용자 확인. ⇒ 평점을 남긴 책이 5000개가 넘는 것을 확인함. 이런 유저들이 소수 있는데, 특정 시설에서 대량 구매한 것이 아닌가

이런 유저들의 나이를 0 or 100 이상으로 처리하여 새로운 타입의 유저로 처리하려고 함.

 하지만 실제로 시설에서 구매했는지 개인이 모두 읽었는지 확인할 방법이 없어서 pass

### (가설 4. age와 rating에 선형관계가 있을 것이다.)

![image](https://user-images.githubusercontent.com/77721219/233974664-4a6ed809-0c95-4669-8b5d-64ebc4bd5140.png)

관련없음^^

### (가설 5. age는 language와 관련있을 것이다.)

![image](https://user-images.githubusercontent.com/77721219/233974727-d4aa13c6-611b-497f-8336-2882a37c3e96.png)

context_data.py의 경우 age 변수를 10살 단위로 나누어 임베딩한다. 그렇기 때문에 language boxplot의 Q3-Q1이 age = 10을 넘지 않는 경우, 해당 language의 책의 평점을 남긴 유저의 나이를 그 boxplot의 평균으로 처리하려고 했다.

하지만 범위가 10이 넘는, en의 책이 압도적으로 많음.

en,fr,de에 해당하는 유저의 경우, 다른 변수와 age와의 상관관계를 또 고려해야함.

그래서 pass.

나머지 age 결측치:

age가 nan인 유저들이 평점을 남긴(구매한) 책들 중 가장 많이 구매한 category를 구한 후. 해당 category의 평균 age로 채웠다.

location은 현석님 방법으로 전처리함.

# 2. Books

books 데이터셋은 변수가 대부분 범주형이어서 따로 시각화를 이용한 EDA는 진행하지 않았다.

### 1. author

(1) 5번째 줄에 ‘;Katie Stewart”’ 와 같이 이름 앞 뒤로 불필요한 특수문자가 있는 경우 모두 제거해주었다.

![image](https://user-images.githubusercontent.com/77721219/233974816-9be24536-fecd-408c-bede-9684f08caee3.png)

(2) 다음과 같이 같은 작가인데 중간에 comma가 들어간 author가 있어 comma를 모두 제거하였다.
![image](https://user-images.githubusercontent.com/77721219/233974948-fb7de38d-0925-4cf0-a73c-d896dfd50119.png)

(3) 이름 중간에 ‘-’ 를 제거.

(4) 같은 작가인데 대소문자 표기가 다른 데이터가 있어, 모두 lowercase로 변경했다.

### 2. language

성현님 방법으로 전처리.

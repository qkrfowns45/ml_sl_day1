# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 19:53:52 2021

@author: qkrfo
"""

#사이킷 런-파이썬 머신러닝 라이브러리 중 가장 많이 사용되는 라이브러리. 파이썬 기반 머신러닝은 곧 사이킷런으로 개발한다고 말 할 정도로 정말 많이 사용함 최근에는 텐서플로, 케라스 등 딥러닝 전문 라이브러리의 강세로 관심 줄어들지만 많은 데이터 분석가가 의존하는 대표적인 파이썬 ML 라이브러리이다

#머신러닝을 위한 매우 다양한 알고리즘과 개발을 위한 편리한 프레임워크 API를 제공한다. 
#아나콘다를 설치하면 자동으로 사이킷런이 설치가 된다.

#datasets는 자체적으로 제공하는 데이터 세트를 생성하는 모듈의 모음
#.tree는 트리 기반 ML알고리즘을 구현한 클래스 모음
#의사결정 알고리즘을 구현하기 위해 DecisionTreeClassifier을 사용 붓꽃 데이터 셋은 load_iris을 사용
import sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
import pandas as pd
from sklearn.metrics import accuracy_score
#사이킷런 버전확인
print(sklearn.__version__)

#사리컷런으로 머신러닝 만들어 보기 - 붓꽃 품종 예측하기

#붓꽃 데이터 세트 로딩
iris = load_iris()

#iris.data는 Iris 데이터 세트에서 피처만으로 된 데이터를 numpy로 가지고 있다.
iris_data = iris.data

#iris.target은 붓꽃 데이터 세트에서 레이블 데이터를 numpy로 가지고 있다.
iris_label = iris.target
#0,1,2,의 데이터값을 가지고 있으며 품종 마다 이름은 target_names로 구분되어 있다. 순서대로 0,1,2의 값을 가지고 있다.
print('iris target값:',iris_label)
print('iris target명:',iris.target_names)

#붓꽃 데이터 세트를 자세히 보기 위해 DataFrame으로 변환합니다.
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label']=iris.target
iris_df.head(3)
print(iris_df)

#학습용 데이터와 테스트용 데이터는 반드시 분리해야 한다. 학습 데이터로는 학습된 모델이 얼마나 뛰어난 성능을 가지는지 평가하려면 테스트 데이터 세트가 필요하기 때문이다. 이를 위해서는 사이킷런이 spilt apt를 제공한다.
#1번째 파라미터는 피처 데이터 세트, 두번째는 레이블 데이터 세트, 그리고 세번째는 전체 데이터 세트 중 테스트 데이터 세트의 비율, 마지막은 호풀할때마다 같은 데이터세트를 생성하기 위한 난수 발생값 spilt함수 호출시 무작위로 데이터를 분리하므로 수행할때마다 다른 데이터를 만들 수 있다.
#실습용 예제이므로 수행할 때마다 동일한 데이터 세트로 분리하기 이ㅜ해 일정한 숫자 값으로 부여
X_train,X_test,y_train,y_test = train_test_split(iris_data,iris_label,test_size = 0.2,random_state=11)

#데이터를 확보했으니 이 데이터를 기반으로 머신러닝 분류 알고리즘의 하나인 의사 결정 트리를 이용해 학습과 예측을 수행해 본다. 먼저 사이킷런 의사 결정 트리인 DecisionTreeClassirier를 객체로 생성
df_clf = DecisionTreeClassifier(random_state=11)
#학습 수행
df_clf.fit(X_train,y_train)
#예측을 수행한다. 예측은 반드시 학습 데이터가 아닌 다른 데이터를 이용해야 하며, 일반적으로 테스트 데이터 세트를 이용힌디/
#학습이 완료된 DecisionTreeClassifier 객체에서 테스트 데이터 세트로 예측 수행
pred = df_clf.predict(X_test)
#성능을 평가한다. 정확도 측정은 사이킷런 메트릭스에서 accuracy_score에서 함수를 제공한다.
print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test, pred)))



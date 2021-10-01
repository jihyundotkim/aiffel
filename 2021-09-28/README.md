# 코드 설명

데이터셋은 3가지이지만 데이터를 로드하여 학습시키고 테스트하는 과정은 거의 비슷하기 때문에 유방암 데이터셋만 따로 설명하도록 하겠습니다.

```python
from sklearn.datasets import load_breast_cancer
```
데이터셋을 `import`합니다.
```python
from sklearn.model_selection import train_test_split
```

데이터셋을 학습 데이터와 테스트 데이터로 나누기 위해 `train_test_split`을 `import`합니다.

```python
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
```
전반적인 지표를 한눈에 보기 위한 `classification_report`와 데이터셋에 적절한 평가 지표를 `import`합니다. 평가 지표 선택에 대한 부분은 아래에 좀 더 자세히 설명하였습니다.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
```
5가지 classifier 모델을 `import`합니다.

```python
import re
```
정규표현식을 이용해 각 모델의 이름을 출력해주기 위해 `re`클래스를 `import`했습니다.

```python
breast_cancer = load_breast_cancer()
```
데이터셋을 로드해줍니다.

```python
features = breast_cancer.data
labels = breast_cancer.target
```

데이터셋의 feature들과 정답 라벨을 변수에 할당합니다.

```python
print(breast_cancer.target_names)
print(breast_cancer.DESCR)
```

노드에 `target_names`를 출력하고 데이터를 describe해보라는 내용이 있어서 일단 했습니다. `target_names`는 \['malignant' 'benign'\]으로 암과 암이 아닌 것 두가지로 분류되는 것을 알 수 있습니다.

```python
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=7)
```

`features`와 `labels`에 저장된 데이터와 정답 라벨 세트를 학습용, 테스트용으로 8:2로 나눕니다. 데이터가 랜덤하게 나뉘기 때문에 매번 같은 조건에서 실행하기 위해서 `random_state`를 지정해줍니다.

```python
decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier()
svm_model = svm.SVC()
sgd = SGDClassifier()
logistic = LogisticRegression(max_iter=10000)
```
Decision Tree, Random Forest, Support Vector, Stochastic Gradient Descent, 그리고 Logistic Regression 모델들의 인스턴스들을 생성해 줍니다.

```python
models_array = [decision_tree, random_forest, svm_model, sgd, logistic]
```
`for`반복문으로 쉽게 5가지 모델들을 다 학습시키고 테스트할 수 있도록 리스트에 넣었습니다.

```python
for model in models_array:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = recall_score(y_test, y_pred) #False Negative가 적은 게 중요한 암 진단이기 때문에 Recall을 사용하였습니다.
    classifier_name = re.search(r'\.[A-Z][a-zA-z]+\'',str(type(model)))
    print(classifier_name.group()[1:-1])
    print(classification_report(y_test, y_pred))
    print("Recall:", score)
```

`models_array`에 있는 각각의 모델들에 대해 `model.fit(X_train, y_train)`으로 학습을 시킵니다. 학습이 된 후 `model_predict(X_test)`로 테스트셋에 대해 예측을 하여 `y_pred`변수에 할당해 줍니다. 그 다음 적절한 평가 지표(여기서는 `recall_score`)를 계산합니다. 어떤 출력 결과가 어떤 모델에서 나온 건지 보기 위해 정규표현식으로 model의 class 이름을 출력한 뒤 테스트 결과를 확인해 보기 위해 `classification_report`와 평가 지표 점수를 출력합니다. 

# 평가 지표 선택

`sklearn.metrics`에는 다양한 평가 함수들이 내장되어 있습니다. 데이터셋의 분포와 목적에 따라 적절한 평가 지표를 적용해야 합니다. 예컨대 false가 많고 true는 적은 데이터셋이라면 전부 false로만 예측해도 정확도 자체는 높게 나오기 때문에 오해가 생길 수 있습니다. 예를 들어 암에 걸리지 않은 사람이 암에 걸린 사람보다 많은 경우겠죠. 그리고 질병을 진단하는 경우라면 병에 걸렸는데 건강하다고 판단하여 치료가 늦어지는 상황을 피하는 것이 최우선이기에 false negative에 중점을 두는 recall을 사용하는 것이 적절할 것입니다.

반면에 붓꽃 종류와 와인 종류는 데이터가 비교적 고르게 분포되어 있고, true, false의 binary가 아닌 multilabel 분류이기 때문에 accuracy가 적절하다고 판단하였습니다.

# 회고

**어려웠던 점**: 노드에 설명이 잘 되어 있어 딱히 프로젝트 자체에서 어려웠던 부분은 없습니다.

**알아낸 점**: sklearn의 다양한 classifier 클래스들을 사용하고 metrics를 통해 평가하는 방법을 배웠습니다. metrics의 종류와 계산방법에 대해서도 배웠습니다.

**아직 모호한 부분**: 각 classifier의 메카니즘을 정확하게 이해하지 못한 채 `import`만 해서 사용한 것 같아 이론 공부를 더 해야겠습니다.

**지표를 맞추기 위해 시도한 것들**: 5가지 Classifier를 적용해 보았고, 데이터셋에 적절한 평가지표는 무엇일까 고민해보았습니다.

**다짐**: Classifier 알고리즘과 `sklearn.metrics`를 더 자세하게 공부해 보아야 하겠습니다.

# 참고한 자료
* [Understanding Data Science Classification Metrics in Scikit-Learn in Python](https://towardsdatascience.com/understanding-data-science-classification-metrics-in-scikit-learn-in-python-3bc336865019)
* [scikit-learn documentation - 3.3. Metrics and scoring: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics)

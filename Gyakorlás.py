#Fibonacci sorozat
def fib(n):
     a, b = 0, 1
     while a < n:
         print(a, end=' ')
         a, b = b, a+b
     print()
fib(100)

#-----------------------------------

#beszélgetés
# 1
print("Szia!")
# válasz
name = input('Hogy hívnak?\n')
print(f'Szia, {name}.')
# 2
print("Hány éves vagy?")
# válasz
kor = input("")
print("Férfi vagy Nő?")
nem = input()
print(f'Amit eddig tudok rólad: {name} a neved, és {kor} éves {nem} vagy.')


-----------------------------------------

#Gépi tanulás TOP ML-ek
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Adat betöltése
data = load_iris()
X = data.data
y = data.target

# Tanító és teszt halmaz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Naiv Bayes

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Naive Bayes pontosság:", accuracy_score(y_test, pred))

# Logisztikus regresszió

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Logisztikus regresszió pontosság:", accuracy_score(y_test, pred))

# Döntés fa

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Döntési fa pontosság:", accuracy_score(y_test, pred))

# KNN legközelebbi szomszéd

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("KNN pontosság:", accuracy_score(y_test, pred))

# Random forrest

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Random Forest pontosság:", accuracy_score(y_test, pred))

# Support vector machine

from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("SVC pontosság:", accuracy_score(y_test, pred))

# XG boost

from xgboost import XGBClassifier

model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("XGBoost pontosság:", accuracy_score(y_test, pred))

# Soft voting classifier

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

model = VotingClassifier(
estimators=[
('lr', LogisticRegression(max_iter=1000)),
('svc', SVC(probability=True)),
('dt', DecisionTreeClassifier())
],
voting='soft' # valószínűségekkel szavaz
)

model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Soft Voting pontosság:", accuracy_score(y_test, pred))


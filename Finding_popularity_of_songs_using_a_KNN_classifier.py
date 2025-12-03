import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#importing data and exploratory data analysis to prepare data for the model
songs = pd.read_csv("Top-50-musicality-global.csv")
pop_scores = []

for i in songs["Popularity"]:
    if i > 80:
        pop_scores.append("Very popular")
    elif i > 60:
        pop_scores.append("Popular")
    elif i > 40:
        pop_scores.append("Middle popularity")
    else:
        pop_scores.append("Unpopular")

songs["Popularity rank"] = pop_scores

print(songs[["Energy", "Liveness", "Acousticness", "Popularity", "Popularity rank"]].head())

# making model, as well as setting a for loop to itterate through different neighbor values
X = songs[["Energy", "Liveness", "Acousticness"]]
y = songs["Popularity rank"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=17, stratify=y)

n_range = np.arange(3, 20)

train_accuracies = []
test_accuracies = []

for z in n_range:
    knn = KNeighborsClassifier(n_neighbors=z)
    knn.fit(X_train, y_train)
    train_accuracies.append(knn.score(X_train, y_train))
    test_accuracies.append(knn.score(X_test, y_test))

#Making a graph, to show the training and testing accuracy for different neighbor values
fig, ax = plt.subplots()
plt.plot(n_range, train_accuracies, label = "Training accuracies")
plt.plot(n_range, test_accuracies, label = "Testing accuracies")
plt.legend()
plt.title("KNN classifier with different n. of neighbors")
plt.xlabel("Number of neighbors")
plt.ylabel("Accuracy")

plt.show()
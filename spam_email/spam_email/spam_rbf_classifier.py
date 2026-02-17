import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from rbf_model import RBF

for folder in ['distribution', 'classifier']:
    if not os.path.exists(folder):
        os.makedirs(folder)

data = pd.read_csv("spam.csv", encoding="latin-1")
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

encoder = LabelEncoder()
y = encoder.fit_transform(data['label'])

vectorizer = TfidfVectorizer(stop_words='english', max_features=400)
X = vectorizer.fit_transform(data['message']).toarray()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

k = 12
kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
kmeans.fit(X_train)
centers = kmeans.cluster_centers_

distances = []
for i in range(len(centers)):
    for j in range(len(centers)):
        if i != j:
            distances.append(np.linalg.norm(centers[i] - centers[j]))
sigma = np.mean(distances)

rbf = RBF(centers, sigma)
rbf.fit(X_train, y_train)

y_pred = rbf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nAccuracy: {acc:.4f}")
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(8, 5))
correct = y_test == y_pred
incorrect = y_test != y_pred

plt.scatter(X_test[correct, 0], X_test[correct, 1], c='green', label='Correct', alpha=0.4)
plt.scatter(X_test[incorrect, 0], X_test[incorrect, 1], c='red', label='Incorrect', alpha=0.8, marker='x')

plt.xlabel("TF-IDF Feature 1")
plt.ylabel("TF-IDF Feature 2")
plt.title(f"RBF Classification Results (Acc: {acc:.2f})")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig('classifier/classifier_results.png')
plt.close()
print("📁 Saved to classifier/classifier_results.png")

labels, counts = np.unique(y, return_counts=True)

plt.figure(figsize=(6, 4))
plt.bar(['Not Spam', 'Spam'], counts, color=['skyblue', 'salmon'], alpha=0.8)
plt.title("Dataset Distribution")
plt.ylabel("Number of Messages")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig('distribution/dataset_distribution.png')
plt.close()
print("📁 Saved to distribution/dataset_distribution.png")

print("\n✅ All set! Files organized.")
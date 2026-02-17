import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from rbf_model import RBF

folder_name = "spam_or_not"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

data = pd.read_csv("spam.csv", encoding="latin-1")
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

encoder = LabelEncoder()
y = encoder.fit_transform(data['label'])

vectorizer = TfidfVectorizer(stop_words='english', max_features=400)
X = vectorizer.fit_transform(data['message']).toarray()

kmeans = KMeans(n_clusters=10, random_state=42, n_init='auto')
kmeans.fit(X)
centers = kmeans.cluster_centers_

dists = []
for i in range(len(centers)):
    for j in range(len(centers)):
        if i != j:
            dists.append(np.linalg.norm(centers[i] - centers[j]))
sigma = np.mean(dists)

rbf = RBF(centers, sigma)
rbf.fit(X, y)

example_message = "Congratulations! You won a free voucher. Click here now!"

vector = vectorizer.transform([example_message]).toarray()
prediction = rbf.predict(vector)[0]
result = "SPAM" if prediction == 1 else "NOT SPAM"

# Print to console
print("-" * 30)
print(f"📩 Testing: {example_message}")
print(f"🔍 Result: {result}")
print("-" * 30)

plt.figure(figsize=(10, 6))

plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.2, label="Dataset", cmap='coolwarm')

plt.scatter(vector[0, 0], vector[0, 1], c='red', s=250, marker='X', 
            edgecolors='black', label=f"Tested: {result}")

plt.title("Email Position in Feature Space", fontsize=14, pad=20)
plt.suptitle(f"Message: \"{example_message}\"", fontsize=10, color='darkblue', y=0.92)

plt.xlabel("TF-IDF Feature 1")
plt.ylabel("TF-IDF Feature 2")
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)

output_path = os.path.join(folder_name, 'spam_test_result.png')

plt.savefig(output_path, bbox_inches='tight')
plt.close() 

print(f"✅ Success! Visualization saved in '{output_path}'.")
print("The script has finished and closed itself.")
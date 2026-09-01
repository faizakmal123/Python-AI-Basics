"""
Simple K-Nearest Neighbors (KNN) Classifier from Scratch
Eksplorasi fundamental algoritma Machine Learning dan klasifikasi pola.
"""

import math

def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

class SimpleKNN:
    def __init__(self, k=3):
        self.k = k
        self.dataset = []

    def fit(self, features, labels):
        self.dataset = list(zip(features, labels))

    def predict(self, sample):
        distances = [
            (euclidean_distance(sample, features), label)
            for features, label in self.dataset
        ]
        distances.sort(key=lambda item: item[0])
        nearest_labels = [label for _, label in distances[:self.k]]
        return max(set(nearest_labels), key=nearest_labels.count)

if __name__ == "__main__":
    # Dataset latihan: [Fitur 1, Fitur 2] -> Label
    X_train = [
        [1.0, 1.2], [1.1, 0.9], [0.9, 1.0],   # Cluster A: Normal
        [5.0, 5.1], [5.2, 4.9], [4.8, 5.0]    # Cluster B: Anomaly
    ]
    y_train = ["Normal", "Normal", "Normal", "Anomaly", "Anomaly", "Anomaly"]

    model = SimpleKNN(k=3)
    model.fit(X_train, y_train)

    test_samples = [[1.1, 1.0], [4.9, 5.2]]
    print("--- KNN Pattern Classification ---")
    for sample in test_samples:
        pred = model.predict(sample)
        print(f"Input: {sample} -> Classified as: {pred}")

import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.resize(img, (128, 128))
    b_avg, g_avg, r_avg, _ = cv2.mean(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    return [r_avg, g_avg, b_avg, brightness]

path_dataset = 'dataset_sampah'
categories = ['anorganik', 'organik']

X = []
y = []

print("Proses mengekstrak fitur gambar...")
for category in categories:
    folder_path = os.path.join(path_dataset, category)
    label = categories.index(category)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        features = extract_features(img_path)

        if features:
            X.append(features)
            y.append(label)

def hitung_entropy(y):
    elemen, hitung = np.unique(y, return_counts=True)
    probabilitas = hitung / len(y)
    entropy = -np.sum(probabilitas * np.log2(probabilitas))
    return entropy

def information_gain(y, y_kiri, y_kanan):
    p_kiri = len(y_kiri) / len(y)
    p_kanan = len(y_kanan) / len(y)
    gain = hitung_entropy(y) - (p_kiri * hitung_entropy(y_kiri) + p_kanan * hitung_entropy(y_kanan))
    return gain

class Node:
    def __init__(self, feature=None, threshold=None, kiri=None, kanan=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.kiri = kiri
        self.kanan = kanan
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            nilai_terbanyak = np.round(np.mean(y))
            return Node(value=int(nilai_terbanyak))

        best_feat, best_thresh = self._best_split(X, y, n_features)

        idx_kiri = X[:, best_feat] <= best_thresh
        idx_kanan = X[:, best_feat] > best_thresh

        kiri = self._grow_tree(X[idx_kiri], y[idx_kiri], depth + 1)
        kanan = self._grow_tree(X[idx_kanan], y[idx_kanan], depth + 1)

        return Node(feature=best_feat, threshold=best_thresh, kiri=kiri, kanan=kanan)

    def _best_split(self, X, y, n_features):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thresh in thresholds:
                idx_kiri = X_column <= thresh
                idx_kanan = X_column > thresh

                if len(y[idx_kiri]) == 0 or len(y[idx_kanan]) == 0:
                    continue

                gain = information_gain(y, y[idx_kiri], y[idx_kanan])

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh

        return split_idx, split_thresh

    def prediksi(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.kiri)
        return self._traverse_tree(x, node.kanan)

class RandomForest:
    def __init__(self, n_estimators = 10, max_depth=5, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_bootstrap, y_bootstrap)

            self.trees.append(tree)

    def prediksi(self, X):
        tree_preds = np.array([tree.prediksi(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [int(np.round(np.mean(pred_per_gambar))) for pred_per_gambar in tree_preds]

        return np.array(y_pred)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTree(max_depth=5)
clf.fit(X_train, y_train)

clf_rf = RandomForest(n_estimators=20, max_depth=5)
clf_rf.fit(X_train, y_train)

print(f"\nTotal seluruh dataset : {len(X)}")
print(f"Total training dataset : {len(X_train)}")
print(f"Total testing dataset : {len(X_test)}")

print(f"\nData Train Anorganik : {len(y_train[y_train == 0])}")
print(f"Data Train Organik : {len(y_train[y_train == 1])}")

y_pred = clf.prediksi(X_test)
akurasi = np.mean(y_pred == y_test) * 100
print(f"\nAkurasi Decision Tree: {akurasi:.2f}%")

benar = (y_pred == y_test)
prediksi_benar = np.sum(benar)
prediksi_salah = len(benar) - prediksi_benar
print(f'Data prdiksi benar : {prediksi_benar}')
print(f'Data prediksi salah : {prediksi_salah}')

y_pred_rf = clf_rf.prediksi(X_test)
akurasi_rf = np.mean(y_pred_rf == y_test) * 100
print(f"\nAkurasi Random Forest: {akurasi_rf:.2f}%")

benar = (y_pred_rf == y_test)
prediksi_benar = np.sum(benar)
prediksi_salah = len(benar) - prediksi_benar
print(f'Data prdiksi benar : {prediksi_benar}')
print(f'Data prediksi salah : {prediksi_salah}')

def tes_prediksi(file_path):
    feat = extract_features(file_path)
    if feat:
        pred = clf.prediksi([feat])
        print(f"\nMenggunakan Decision Tree,")
        print(f"File {file_path} diprediksi sebagai: {categories[pred[0]]}")

def tes_prediksi2(file_path):
    feat = extract_features(file_path)
    if feat:
        pred = clf_rf.prediksi([feat])
        print(f"\nMenggunakan Random Forest,")
        print(f"File {file_path} diprediksi sebagai: {categories[pred[0]]}")

tes_prediksi('pisang.jpeg')
tes_prediksi2('pisang.jpeg')
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# --------------------------
# 1. Загрузка эмбеддингов
# --------------------------
file_path = 'embeddings.pkl'  # путь к вашему файлу

with open(file_path, 'rb') as f:
    X, y = pickle.load(f)  # X — эмбеддинги, y — метки людей

print("Форма эмбеддингов:", X.shape)
print("Форма меток:", y.shape)

# --------------------------
# 2. Уменьшение размерности до 3D с помощью PCA
# --------------------------
pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(X)

# --------------------------
# 3. Создание 3D графика
# --------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Получаем уникальные метки (каждого человека)
unique_labels = np.unique(y)
colors = plt.cm.get_cmap('tab10', len(unique_labels))  # палитра для цветов

# Рисуем каждую группу людей своим цветом
for i, label in enumerate(unique_labels):
    indices = y == label
    ax.scatter(
        embeddings_3d[indices, 0],
        embeddings_3d[indices, 1],
        embeddings_3d[indices, 2],
        c=[colors(i)],
        label=f'Person {label}',
        s=50
    )

# --------------------------
# 4. Настройка подписей и легенды
# --------------------------
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.set_title('3D визуализация эмбеддингов лиц')
ax.legend()

# --------------------------
# 5. Показ графика
# --------------------------
plt.show()

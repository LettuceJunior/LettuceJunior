import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import KMeans

# Зчитування датасету з файлу
dataset_path = 'DS3.txt'
data = np.loadtxt(dataset_path)

# Кількість кластерів для KMeans
num_clusters = 5

# Застосування KMeans для знаходження зв'язаних областей
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(data)
labels = kmeans.labels_

# Знаходження центрів ваги зв'язаних областей
cluster_centers = []
for i in range(num_clusters):
    cluster_points = data[labels == i]
    center = np.mean(cluster_points, axis=0)
    cluster_centers.append(center)

cluster_centers = np.array(cluster_centers)

# Побудова діаграми Вороного
vor = Voronoi(cluster_centers)

# Створення візуалізації
fig, ax = plt.subplots(figsize=(9.6, 5.4))

# Відображення центрів ваги
ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='o', s=100)

# Відображення діаграми Вороного
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', line_width=2, line_alpha=0.6, point_size=0)

# Відображення точок вихідного датасету
ax.scatter(data[:, 0], data[:, 1], c='black', s=5, alpha=0.1)

# Встановлення розмірів вікна
ax.set_xlim(0, 960)
ax.set_ylim(0, 540)
ax.set_aspect('equal', adjustable='box')

# Збереження результату у файл графічного формату
plt.savefig('результат.png')

# Відображення графіка
plt.show()

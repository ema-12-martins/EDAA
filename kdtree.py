import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Classe para representar um nó da KDTree
class KDNode:
    def __init__(self, point, index, left=None, right=None):
        self.point = point
        self.index = index
        self.left = left
        self.right = right

# Função para construir a árvore
def build_kdtree(points, depth=0):
    if not points:
        return None

    k = len(points[0][0])  # número de dimensões
    axis = depth % k
    points.sort(key=lambda x: x[0][axis])
    median = len(points) // 2

    return KDNode(
        point=points[median][0],
        index=points[median][1],
        left=build_kdtree(points[:median], depth + 1),
        right=build_kdtree(points[median + 1:], depth + 1)
    )

# Função de busca KNN
def knn_search(root, target, k, depth=0, heap=None):
    import heapq
    if heap is None:
        heap = []

    if root is None:
        return heap

    axis = depth % len(target)
    dist = np.linalg.norm(np.array(target) - np.array(root.point))
    heapq.heappush(heap, (-dist, root.index, root.point))

    if len(heap) > k:
        heapq.heappop(heap)

    diff = target[axis] - root.point[axis]
    close, away = (root.left, root.right) if diff < 0 else (root.right, root.left)

    knn_search(close, target, k, depth + 1, heap)

    if len(heap) < k or abs(diff) < -heap[0][0]:
        knn_search(away, target, k, depth + 1, heap)

    return sorted([(-d, idx) for d, idx, _ in heap])


# ======================== USO =========================

# 1. Lê o CSV
df = pd.read_csv('./archive/fashion-dataset/styles.csv', quotechar='"', on_bad_lines='skip', encoding='utf-8')

# 2. Seleciona colunas categóricas
columns_to_use = ['gender', 'masterCategory', 'subCategory', 'baseColour', 'season', 'usage']
X_raw = df[columns_to_use].fillna('missing')

# 3. Codifica com OneHotEncoder
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X_raw).toarray()

# 4. Constrói lista de pontos [(ponto, índice)]
points = [(X_encoded[i], i) for i in range(len(X_encoded))]

# 5. Constrói a árvore manualmente
tree = build_kdtree(points)

# 6. Busca os 5 mais próximos do primeiro item
neighbors = knn_search(tree, X_encoded[0], k=5)

# 7. Mostra resultados
print("Vizinhos mais próximos:")
for dist, idx in neighbors:
    print(f"Distância: {dist:.4f}, Índice: {idx}")
    print(df.iloc[idx])

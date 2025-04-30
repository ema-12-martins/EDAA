import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import heapq

# Mapeamento das cores que combinam, para efeito de exemplo, você pode expandir esse mapeamento
color_combinations = {
    'Black': ['White', 'Grey', 'Beige', 'Red', 'Silver', 'Gold', 'Navy Blue'],
    'White': ['Black', 'Grey', 'Beige', 'Red', 'Silver', 'Gold', 'Navy Blue'],
    'Grey': ['Black', 'White', 'Beige', 'Navy Blue', 'Red', 'Silver'],
    'Blue': ['Navy Blue', 'White', 'Grey', 'Red', 'Beige', 'Black'],
    'Red': ['Black', 'White', 'Grey', 'Blue', 'Beige', 'Gold'],
    'Green': ['Black', 'White', 'Grey', 'Beige', 'Navy Blue'],
    'Navy Blue': ['Black', 'White', 'Grey', 'Beige', 'Red', 'Silver', 'Gold'],
    # Adicionar mais combinações de cores conforme necessário
}

# Função para atribuir peso às cores baseado na cor do produto base
def calculate_color_weight(base_color, product_color):
    # Se a cor do produto for nula (na), atribui peso 1
    if product_color == 'nan' or base_color == 'nan':
        return 1
    
    # Caso a cor seja a mesma
    if base_color == product_color:
        return 2  # Peso alto para cor exata
    
    # Caso as cores combinam de acordo com o mapeamento
    if product_color in color_combinations.get(base_color, []):
        return 1.5  # Peso intermediário para cores que combinam
    
    # Caso não haja combinação
    return 1  # Peso 1 para outras cores

# Função para atribuir pesos a todas as cores de um produto com base na cor base
def assign_weights_based_on_color(df, base_color_column='baseColour'):
    weights = []
    
    # Itera sobre o dataframe para calcular os pesos
    for _, row in df.iterrows():
        product_color = row[base_color_column]
        weight = calculate_color_weight(base_color_column, product_color)
        weights.append(weight)
        
    return np.array(weights)

# ======================== CLASSE KDTREE ==========================

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

# Função de busca KNN com distância Euclidiana ponderada
def knn_search(root, target, k, weights=None, depth=0, heap=None):
    if heap is None:
        heap = []
        
    if weights is None:
        weights = np.ones(len(target))  # Se não for passado, todos os pesos são 1 (sem peso extra)

    if root is None:
        return heap

    axis = depth % len(target)
    
    # Calcula a distância Euclidiana ponderada
    dist = np.sqrt(np.sum(weights * (np.array(target) - np.array(root.point))**2))
    
    # Adiciona o nó ao heap com a distância negativa (para que o heap funcione como uma fila de prioridade)
    heapq.heappush(heap, (-dist, root.index, root.point))

    if len(heap) > k:
        heapq.heappop(heap)

    diff = target[axis] - root.point[axis]
    close, away = (root.left, root.right) if diff < 0 else (root.right, root.left)

    knn_search(close, target, k, weights, depth + 1, heap)

    if len(heap) < k or abs(diff) < -heap[0][0]:
        knn_search(away, target, k, weights, depth + 1, heap)

    return sorted([(-d, idx) for d, idx, _ in heap])

# ======================== USO =========================

#Numero da linha do produto a procurar
def get_recommendations(id):

    # 1. Lê o CSV
    df = pd.read_csv('./archive/fashion-dataset/styles.csv', quotechar='"', on_bad_lines='skip', encoding='utf-8')
    product_index = 0
    try:
        # Find the index of the row where the 'id' column matches the given product_id
        product_index = df[df['id'] == id].index[0]
    except IndexError:
        print(f"Product ID {id} not found in the dataset.")
    
    print("Produto original a procurar:")
    print(df.iloc[product_index].id)

    # 2. Seleciona colunas categóricas
    columns_to_use = ['gender', 'masterCategory', 'subCategory', 'baseColour', 'season', 'usage']
    X_raw = df[columns_to_use].fillna('missing')

    # 3. Codifica com OneHotEncoder
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X_raw).toarray()

    # 4. Divisão das colunas numéricas (exemplo para a coluna 'price') em 5 casos
    # Vamos supor que você tenha uma coluna numérica chamada 'price' no seu dataframe
    # Você pode substituir 'price' por qualquer coluna numérica relevante que você tenha
    if 'price' in df.columns:
        df['price_case'] = pd.cut(df['price'], bins=5, labels=[f'Case {i+1}' for i in range(5)])
        price_encoded = encoder.fit_transform(df[['price_case']]).toarray()
        X_encoded = np.hstack([X_encoded, price_encoded])

    # 5. Constrói lista de pontos [(ponto, índice)]
    points = [(X_encoded[i], i) for i in range(len(X_encoded))]

    # 6. Constrói a árvore manualmente
    tree = build_kdtree(points)

    # 7. Definindo pesos (dando mais peso à subCategoria e baseCor)
    # Primeiro, identifique o número de categorias para subCategory e baseColour
    subCategory_start = X_encoded.shape[1] - len(encoder.categories_[2])  # SubCategory começa depois das duas primeiras colunas
    baseColour_start = subCategory_start + len(encoder.categories_[2])  # BaseColour começa depois da subCategory
    usage_start = (
        len(encoder.categories_[0]) +  len(encoder.categories_[1]) + len(encoder.categories_[2]) + len(encoder.categories_[3]) + 
        len(encoder.categories_[4]))

    weights = np.ones(X_encoded.shape[1])  # Inicia todos com peso 1

    # Atribuindo mais peso para subCategory e baseColour (peso 2)
    weights[subCategory_start:subCategory_start + len(encoder.categories_[2])] = 2  # Peso 2 para subCategory
    weights[baseColour_start:baseColour_start + len(encoder.categories_[3])] = 2  # Peso 2 para baseColour
    weights[usage_start:usage_start + len(encoder.categories_[5])] = 2 # Peso 2 para o usage

    # 8. Busca os 5 mais próximos do item 10
    neighbors = knn_search(tree, X_encoded[product_index], k=5, weights=weights)

    # 9. Mostra resultados
    print("Vizinhos mais próximos:")
    for dist, idx in neighbors:
        #print(f"Distância: {dist:.4f}, Índice: {idx}")
        #print(df.iloc[idx])  # Mostra o item mais próximo
        print(df.iloc[idx].id)

    id_list = df.iloc[[idx for _, idx in neighbors]].id.tolist()

    recommended_products = [f'{id}.jpg' for id in id_list]    
    return recommended_products

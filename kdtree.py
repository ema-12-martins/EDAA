import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import heapq
import os
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

import pandas as pd
import json
import os

def join_csv_with_json(csv_path, json_dir, output_csv_path):
    df = pd.read_csv(csv_path, quotechar='"', on_bad_lines='skip', encoding='utf-8')

    # Novas colunas a adicionar
    df['price'] = None
    df['discountedPrice'] = None
    df['brandName'] = None
    df['ageGroup'] = None
    df['gender'] = None

    for index, row in df.iterrows():
        product_id = str(row['id'])
        json_path = os.path.join(json_dir, f"{product_id}.json")
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    product_data = data.get('data', {})
                    df.at[index, 'price'] = product_data.get('price')
                    df.at[index, 'discountedPrice'] = product_data.get('discountedPrice')
                    df.at[index, 'brandName'] = product_data.get('brandName')
                    df.at[index, 'ageGroup'] = product_data.get('ageGroup')
                    df.at[index, 'gender'] = product_data.get('gender')
                except json.JSONDecodeError:
                    print(f"Erro ao decodificar JSON para o ID {product_id}")
        else:
            print(f"Arquivo JSON não encontrado para ID {product_id}")

    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"Arquivo enriquecido salvo em: {output_csv_path}")

def add_has_discount_column(file_path):
    df = pd.read_csv(file_path, quotechar='"', on_bad_lines='skip', encoding='utf-8')

    if 'price' in df.columns and 'discountedPrice' in df.columns:
        df['has_discount'] = (df['price'] != df['discountedPrice']).astype(int)
        df.to_csv(file_path, index=False) 
        print(f"Coluna 'has_discount' adicionada com sucesso em '{file_path}'.")
    else:
        print("As colunas 'price' e/ou 'discountedPrice' não estão presentes no DataFrame.")



# ======================== USO =========================

#Numero da linha do produto a procurar
def get_recommendations(id):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    from kdtree import build_kdtree, knn_search  # Certifica-te que estas funções estão definidas

    # 1. Lê o CSV
    df = pd.read_csv('styles_joined.csv', quotechar='"', on_bad_lines='skip', encoding='utf-8')

    # Extrai o ID (remove o ".json")
    id_aux = id.split('.')[0]

    try:
        product_index = df[df['id'] == int(id_aux)].index[0]
    except IndexError:
        print(f"Product ID {id_aux} not found in the dataset.")
        return []
        
    print("Produto original a procurar:")
    print(df.iloc[product_index].id)

    # 2. Colunas categóricas
    columns_to_use = ['gender', 'masterCategory', 'subCategory', 'articleType',
                      'baseColour', 'season', 'usage', 'brandName', 'ageGroup']
    X_raw = df[columns_to_use].fillna('missing')

    # 3. Codifica as categorias
    encoder_cat = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat = encoder_cat.fit_transform(X_raw)

    # 4. Codifica a coluna 'price' em faixas
    if 'price' in df.columns:
        df['price_case'] = pd.cut(df['price'], bins=5, labels=[f'Case {i+1}' for i in range(5)])
        encoder_price = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_price = encoder_price.fit_transform(df[['price_case']])
        X_encoded = np.hstack([X_cat, X_price])
    else:
        X_encoded = X_cat

    # 5. Constrói lista de pontos [(ponto, índice)]
    points = [(X_encoded[i], i) for i in range(len(X_encoded))]

    # 6. Constrói a árvore
    tree = build_kdtree(points)

    # 7. Calcula índices de início de cada coluna
    starts = [0]
    for cats in encoder_cat.categories_[:-1]:
        starts.append(starts[-1] + len(cats))
    col_start_dict = dict(zip(columns_to_use, starts))

    # 8. Define pesos (peso 2 para subCategory, baseColour, usage)
    weights = np.ones(X_encoded.shape[1])
    for col in ['subCategory', 'baseColour', 'usage']:
        start = col_start_dict[col]
        length = len(encoder_cat.categories_[columns_to_use.index(col)])
        weights[start:start+length] = 2
        
    # 8b. Mais peso para produtos com promoção
    weights[-1] = 2

    # 9. Busca os 5 vizinhos mais próximos
    neighbors = knn_search(tree, X_encoded[product_index], k=5, weights=weights)

    # 10. Mostra resultados
    print("Vizinhos mais próximos:")
    for dist, idx in neighbors:
        print(df.iloc[idx].id)

    id_list = df.iloc[[idx for _, idx in neighbors]].id.tolist()
    recommended_products = [f'{id}.jpg' for id in id_list]
    return recommended_products

if __name__ == "__main__":
    # Código para testar individualmente esse módulo
    get_recommendations("1163.json")

#RODAR APENAS 1x, para JUNTAR OS DATASETS
#join_csv_with_json('./fashion-dataset/styles.csv', './fashion-dataset/styles', 'styles_joined.csv') 
#add_has_discount_column('styles_joined.csv'
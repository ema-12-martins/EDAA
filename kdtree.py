import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KDTree

# 1. Lê o CSV
df = pd.read_csv('./archive/fashion-dataset/styles.csv', quotechar='"', on_bad_lines='skip', encoding='utf-8')


# 2. Seleciona as colunas categóricas para a KDTree
columns_to_use = ['gender', 'masterCategory', 'subCategory', 'baseColour', 'season', 'usage']
X_raw = df[columns_to_use].fillna('missing')

# 3. Codifica com OneHotEncoder
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X_raw).toarray()

# 4. Cria a KDTree
tree = KDTree(X_encoded)

# (Opcional) exemplo de consulta: encontra os 5 mais próximos do primeiro item
distances, indices = tree.query([X_encoded[0]], k=5)

# Mostra resultados
print("Distâncias:", distances)
print("Índices:", indices)
print("Itens semelhantes:")
print(df.iloc[indices[0]])

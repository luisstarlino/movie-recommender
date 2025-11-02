import os
import shutil
import kagglehub
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
import pickle

print("📥 Baixando dataset...")

# Baixa e extrai o dataset no cache do root
dataset_path = kagglehub.dataset_download("grouplens/movielens-20m-dataset")
print(f"✅ Dataset baixado em: {dataset_path}")

files = os.listdir(dataset_path)
print("📂 Arquivos disponíveis:", files)

# Corrigido para os nomes corretos do dataset MovieLens 20M
ratings_path = os.path.join(dataset_path, "rating.csv")
movies_path = os.path.join(dataset_path, "movie.csv")

if not (os.path.exists(ratings_path) and os.path.exists(movies_path)):
    raise FileNotFoundError(f"Arquivos rating.csv e movie.csv não encontrados em {dataset_path}")

print("📊 Lendo os arquivos CSV...")
ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path)

print("✅ Dados carregados!")

# --- 🔢 Reindexação dos IDs ---
print("🔢 Reindexando IDs de usuários e filmes...")

unique_users = ratings["userId"].unique()
unique_movies = ratings["movieId"].unique()

user_mapping = {uid: i for i, uid in enumerate(unique_users)}
movie_mapping = {mid: i for i, mid in enumerate(unique_movies)}

# Adiciona colunas com índices contínuos
ratings["user_idx"] = ratings["userId"].map(user_mapping)
ratings["movie_idx"] = ratings["movieId"].map(movie_mapping)

# --- Criação do dataset LightFM ---
dataset = Dataset()
dataset.fit(ratings["user_idx"], ratings["movie_idx"])

(interactions, weights) = dataset.build_interactions(
    (x.user_idx, x.movie_idx, x.rating) for x in ratings.itertuples()
)

print("🧠 Treinando modelo LightFM...")
model = LightFM(loss="warp")
model.fit(interactions, epochs=10, num_threads=4)

# --- Salvando o modelo e os mapeamentos ---
output_dir = "/app/app"
os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(output_dir, "model.pkl")
mapping_path = os.path.join(output_dir, "mappings.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

with open(mapping_path, "wb") as f:
    pickle.dump({"user_mapping": user_mapping, "movie_mapping": movie_mapping}, f)

print(f"✅ Modelo salvo em {model_path}")
print(f"✅ Mapeamentos salvos em {mapping_path}")

# Copia CSVs também, pra facilitar debug
shutil.copy(ratings_path, os.path.join(output_dir, "ratings.csv"))
shutil.copy(movies_path, os.path.join(output_dir, "movies.csv"))

print(f"✅ Arquivos CSV copiados para {output_dir}");

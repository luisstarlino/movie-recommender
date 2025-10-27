import os
import kagglehub
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
import pickle

print("📥 Baixando dataset...")

dataset_path = kagglehub.dataset_download("grouplens/movielens-20m-dataset")
print(f"✅ Dataset baixado em: {dataset_path}")

files = os.listdir(dataset_path)
print("📂 Arquivos disponíveis:", files)

# Corrige os nomes de arquivo conforme o conteúdo real
ratings_path = os.path.join(dataset_path, "rating.csv")
movies_path = os.path.join(dataset_path, "movie.csv")

if not (os.path.exists(ratings_path) and os.path.exists(movies_path)):
    raise FileNotFoundError(f"Arquivos rating.csv e movie.csv não encontrados em {dataset_path}")

print("📊 Lendo os arquivos CSV...")
ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path)

print("✅ Dados carregados!")

dataset = Dataset()
dataset.fit(ratings['userId'], ratings['movieId'])

(interactions, weights) = dataset.build_interactions(
    (x.userId, x.movieId, x.rating) for x in ratings.itertuples()
)

print("🧠 Treinando modelo LightFM...")
model = LightFM(loss='warp')
model.fit(interactions, epochs=10, num_threads=4)

os.makedirs("/app/app", exist_ok=True)
with open("/app/app/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Modelo treinado e salvo em /app/app/model.pkl")

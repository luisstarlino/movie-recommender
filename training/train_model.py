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
    raise FileNotFoundError(f"Arquivos ratingscsv e movie.csv não encontrados em {dataset_path}")

print("📊 Lendo os arquivos CSV...")
ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path)

print("✅ Dados carregados!")

# Criando dataset para LightFM
dataset = Dataset()
dataset.fit(ratings['userId'], ratings['movieId'])

interactions, weights = dataset.build_interactions(
    (x.userId, x.movieId, x.rating) for x in ratings.itertuples()
)

print("🧠 Treinando modelo LightFM...")
model = LightFM(loss='warp')
model.fit(interactions, epochs=10, num_threads=4)

# Garantindo diretório de saída
output_dir = "/app/app"
os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(output_dir, "model.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Modelo treinado e salvo em {model_path}")

shutil.copy(ratings_path, os.path.join(output_dir, "ratings.csv"))
shutil.copy(movies_path, os.path.join(output_dir, "movies.csv"))

print(f"✅ Arquivos CSV copiados para {output_dir}")

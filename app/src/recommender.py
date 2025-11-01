import pickle
import pandas as pd
from lightfm import LightFM
import os
import logging

# === Configuração global de logs ===
logging.basicConfig(
    level=logging.DEBUG,  # Mostra tudo (DEBUG, INFO, WARNING, etc)
    format="%(asctime)s [%(levelname)s] %(message)s",
)

class HybridRecommender:
    def __init__(self):
        logging.info("🔄 Inicializando HybridRecommender...")

        self.model_path = "/app/app/model.pkl"
        self.movies_path = "/app/app/movies.csv"
        self.ratings_path = "/app/app/ratings.csv"

        # === Verifica se o modelo existe ===
        if not os.path.exists(self.model_path):
            logging.error(f"❌ Modelo não encontrado em {self.model_path}")
            raise FileNotFoundError(f"❌ Modelo não encontrado em {self.model_path}")

        with open(self.model_path, "rb") as f:
            self.model: LightFM = pickle.load(f)

        # === Carrega dados ===
        if os.path.exists(self.movies_path):
            self.movies = pd.read_csv(self.movies_path)
            logging.info(f"🎬 Filmes carregados: {len(self.movies)}")
        else:
            logging.warning("⚠️ Nenhum arquivo de filmes encontrado.")
            self.movies = pd.DataFrame(columns=["movieId", "title", "genres"])

        if os.path.exists(self.ratings_path):
            self.ratings = pd.read_csv(self.ratings_path)
            logging.info(f"⭐ Avaliações carregadas: {len(self.ratings)}")
        else:
            logging.warning("⚠️ Nenhum arquivo de ratings encontrado.")
            self.ratings = pd.DataFrame(columns=["userId", "movieId", "rating"])

        logging.info("✅ Modelo e dados carregados com sucesso!")

    # ============================================
    def get_recommendations(self, genre: str, min_popularity: int):
        """Gera recomendações simples baseadas no gênero e popularidade."""
        logging.info(f"🎯 Gerando recomendações para gênero='{genre}', min_popularity={min_popularity}")

        if self.movies.empty:
            logging.warning("⚠️ Nenhum dado de filmes disponível.")
            return pd.DataFrame(columns=["title", "score"])

        # Normaliza gênero
        self.movies["genres"] = self.movies["genres"].fillna("").str.lower()
        genre = genre.lower()

        # Filtra por gênero
        filtered = self.movies[self.movies["genres"].str.contains(genre, na=False)].copy()
        logging.debug(f"🔎 Filmes encontrados no gênero '{genre}': {len(filtered)}")

        if filtered.empty:
            logging.warning(f"Nenhum filme encontrado para o gênero '{genre}'")
            return pd.DataFrame(columns=["title", "score"])

        # Calcula popularidade (contagem de avaliações)
        if not self.ratings.empty:
            popularity = (
                self.ratings["movieId"]
                .value_counts()
                .rename_axis("movieId")
                .reset_index(name="count")
            )
            filtered = filtered.merge(popularity, on="movieId", how="left")
            filtered["count"] = filtered["count"].fillna(0).astype(int)
            filtered = filtered[filtered["count"] >= min_popularity]

            logging.debug(f"📊 Filmes após filtro de popularidade: {len(filtered)}")

        if filtered.empty:
            logging.warning(f"Nenhum filme atingiu o limite mínimo de popularidade {min_popularity}")
            return pd.DataFrame(columns=["title", "score"])

        # Calcula score baseado na popularidade
        filtered["score"] = (filtered["count"] / filtered["count"].max()) * 5
        filtered["score"] = filtered["score"].fillna(0)

        filtered = filtered.sort_values("score", ascending=False)
        top = filtered.head(10)[["title", "score"]]

        logging.info(f"✅ {len(top)} recomendações geradas com sucesso.")
        logging.debug(f"🎬 Top recomendações:\n{top}")

        return top
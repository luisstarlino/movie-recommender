import pickle
import pandas as pd
from lightfm import LightFM
import numpy as np
import os
import logging

# === Configuração global de logs ===
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

class HybridRecommender:
    def __init__(self):
        logging.info("🔄 Inicializando HybridRecommender...")

        self.model_path = "/app/app/model.pkl"
        self.mappings_path = "/app/app/mappings.pkl"
        self.movies_path = "/app/app/movies.csv"
        self.ratings_path = "/app/app/ratings.csv"

        # === Verifica se o modelo existe ===
        if not os.path.exists(self.model_path):
            logging.error(f"❌ Modelo não encontrado em {self.model_path}")
            raise FileNotFoundError(f"❌ Modelo não encontrado em {self.model_path}")

        with open(self.model_path, "rb") as f:
            self.model: LightFM = pickle.load(f)

        # === Carrega mapeamentos ===
        if os.path.exists(self.mappings_path):
            with open(self.mappings_path, "rb") as f:
                mappings = pickle.load(f)
            self.user_mapping = mappings.get("user_mapping", {})
            self.movie_mapping = mappings.get("movie_mapping", {})
            logging.info(f"📑 Mapeamentos carregados: {len(self.user_mapping)} usuários, {len(self.movie_mapping)} filmes")
        else:
            logging.warning("⚠️ Nenhum arquivo mappings.pkl encontrado — predições colaborativas podem falhar.")
            self.user_mapping = {}
            self.movie_mapping = {}

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
    def get_recommendations(self, genre: str, min_popularity: int, min_rating: float):
        """Gera recomendações híbridas (conteúdo + colaborativo)."""
        logging.info(
            f"🎯 Gerando recomendações híbridas para gênero='{genre}', "
            f"min_popularity={min_popularity}, min_rating={min_rating}"
        )

        if self.movies.empty:
            logging.warning("⚠️ Nenhum dado de filmes disponível.")
            return pd.DataFrame(columns=["title", "score"])

        # --- Filtragem de Conteúdo ---
        self.movies["genres"] = self.movies["genres"].fillna("").str.lower()
        genre = genre.lower()
        filtered = self.movies[self.movies["genres"].str.contains(genre, na=False)].copy()
        logging.debug(f"🔎 Filmes encontrados no gênero '{genre}': {len(filtered)}")

        if filtered.empty:
            logging.warning(f"Nenhum filme encontrado para o gênero '{genre}'")
            return pd.DataFrame(columns=["title", "score"])

        # --- Popularidade ---
        if not self.ratings.empty:
            popularity_df = (
                self.ratings["movieId"]
                .value_counts()
                .rename_axis("movieId")
                .reset_index(name="count")
            )
            filtered = filtered.merge(popularity_df, on="movieId", how="left")
            filtered["count"] = filtered["count"].fillna(0).astype(int)
            filtered = filtered[filtered["count"] >= min_popularity]
            logging.debug(f"📊 Filmes após filtro de popularidade: {len(filtered)}")

        # --- Rating médio ---
        if not self.ratings.empty:
            rating_avg = (
                self.ratings.groupby("movieId")["rating"].mean().rename("avg_rating").reset_index()
            )
            filtered = filtered.merge(rating_avg, on="movieId", how="left")
            filtered["avg_rating"] = filtered["avg_rating"].fillna(0)
            filtered = filtered[filtered["avg_rating"] >= min_rating]
            logging.debug(f"⭐ Filmes após filtro de nota mínima ({min_rating}): {len(filtered)}")

        if filtered.empty:
            logging.warning("⚠️ Nenhum filme restou após os filtros aplicados.")
            return pd.DataFrame(columns=["title", "score"])

        # --- Score de Conteúdo (baseado em popularidade) ---
        if "count" in filtered and filtered["count"].max() > 0:
            filtered["content_score"] = (filtered["count"] / filtered["count"].max()) * 5
        else:
            filtered["content_score"] = 2.5  # valor neutro

        # --- Filtragem Colaborativa com LightFM ---
        try:
            logging.info("🧠 Aplicando modelo LightFM (colaborativo)...")

            # Escolhe usuário de exemplo
            sample_user_id = int(self.ratings["userId"].mode()[0])
            user_idx = self.user_mapping.get(sample_user_id, None)
            if user_idx is None:
                raise ValueError(f"Usuário {sample_user_id} não encontrado no mapeamento")

            # Mapeia movieIds -> índices internos
            filtered["movie_idx"] = filtered["movieId"].map(self.movie_mapping)
            filtered = filtered.dropna(subset=["movie_idx"])

            if filtered.empty:
                raise ValueError("Nenhum dos filmes filtrados possui índice no modelo.")

            movie_idx = filtered["movie_idx"].astype(int).tolist()
            user_ids = np.full(len(movie_idx), user_idx)

            # Predição
            scores = self.model.predict(user_ids, movie_idx)

            # Normaliza e combina com conteúdo
            if np.ptp(scores) > 0:
                scores_norm = (scores - scores.min()) / (scores.max() - scores.min()) * 5
            else:
                scores_norm = np.full_like(scores, 2.5)

            filtered["collab_score"] = scores_norm
            filtered["score"] = 0.5 * filtered["content_score"] + 0.5 * filtered["collab_score"]

            logging.info("✅ Recomendação híbrida combinada com sucesso!")

        except Exception as e:
            logging.warning(f"⚠️ Falha na etapa colaborativa: {e}")
            filtered["score"] = filtered["content_score"]

        # --- Garantia final de filtro ---
        filtered = filtered[filtered["avg_rating"] >= min_rating]

        # --- Retorna top 10 ---
        filtered = filtered.sort_values("score", ascending=False)
        top = filtered.head(10)[["title", "score", "avg_rating"]]

        logging.debug(f"🎬 Top recomendações híbridas (>= {min_rating}):\n{top}")
        return top
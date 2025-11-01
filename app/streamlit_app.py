import streamlit as st
import pandas as pd
import io
import logging
from src.recommender import HybridRecommender

# =======================
# Configurações iniciais
# =======================
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

st.title("🎥 Sistema de Recomendação de Filmes (Híbrido)")
st.markdown("Escolha suas preferências e veja recomendações personalizadas!")

# =======================
# Captura de logs em tempo real
# =======================
log_stream = io.StringIO()
handler = logging.StreamHandler(log_stream)
handler.setLevel(logging.DEBUG)

# Remove handlers duplicados (evita logs repetidos)
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[handler],
)

# =======================
# Entradas do usuário
# =======================
genre = st.selectbox(
    "🎭 Selecione o gênero preferido:",
    ["Action", "Comedy", "Drama", "Romance", "Thriller", "Sci-Fi", "Horror", "Animation"]
)

age_group = st.selectbox(
    "👥 Faixa etária:",
    ["18-25", "26-35", "36-50", "50+"]
)

popularity = st.slider("🔥 Popularidade mínima (número de avaliações)", 0, 500, 50)

# =======================
# Painel lateral de logs
# =======================
st.sidebar.header("🧾 Logs do Sistema")
log_box = st.sidebar.empty()

# =======================
# Ação principal
# =======================
if st.button("Gerar Recomendações"):
    st.info("🔄 Gerando recomendações...")

    try:
        recommender = HybridRecommender()
        recommendations = recommender.get_recommendations(genre, popularity)

        # Atualiza logs em tempo real
        log_box.text(log_stream.getvalue())

        if recommendations.empty:
            st.warning("⚠️ Nenhum filme encontrado com os filtros selecionados.")
        else:
            st.success("✅ Recomendações geradas com sucesso!")

            for _, row in recommendations.iterrows():
                cols = st.columns([3, 1])
                with cols[0]:
                    st.markdown(f"**🎬 {row['title']}**")
                with cols[1]:
                    st.markdown(f"⭐ {row['score']:.2f}")

    except Exception as e:
        logging.exception("Erro ao gerar recomendações:")
        log_box.text(log_stream.getvalue())
        st.error(f"❌ Ocorreu um erro: {e}")

# =======================
# Atualiza logs no sidebar sempre
# =======================
log_box.text(log_stream.getvalue())

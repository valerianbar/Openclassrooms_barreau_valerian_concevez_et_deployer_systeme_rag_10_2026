# Image de l'API RAG (FastAPI + FAISS + embeddings locaux + OpenAI)
#
#   Construire : docker build -t rag-openagenda .
#   Lancer     : docker run --rm -p 8000:8000 --env-file .env rag-openagenda
#   Swagger    : http://localhost:8000/docs
#
# Construction en deux étapes : la première installe les dépendances avec uv,
# la seconde ne garde que le résultat (image finale plus légère, sans uv).

# ============================== 1. Dépendances ==============================
FROM python:3.13-slim AS builder

# uv copié depuis son image officielle (version figée pour des builds reproductibles)
COPY --from=ghcr.io/astral-sh/uv:0.9.8 /uv /bin/uv

# UV_COMPILE_BYTECODE : précompile les .py (démarrage plus rapide)
# UV_LINK_MODE=copy    : copie les fichiers (le cache est sur un autre volume)
# UV_PYTHON_DOWNLOADS  : utilise le Python de l'image, sans en télécharger un autre
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Seuls les fichiers de dépendances sont copiés : tant qu'ils ne changent pas,
# Docker réutilise cette couche (pas de réinstallation à chaque modification du code)
COPY pyproject.toml uv.lock .python-version ./

# --locked : versions exactes de uv.lock ; --no-dev : sans notebook ni Ragas
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project


# ============================== 2. Image finale ==============================
FROM python:3.13-slim

# Utilisateur sans privilèges : l'API ne tourne pas en root
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app \
    && chown app:app /app

WORKDIR /app

# Environnement virtuel préparé à l'étape 1
COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1

# À partir d'ici, tout est fait en tant que `app` : les fichiers créés lui appartiennent
# directement (un `chown -R` après coup dupliquerait le modèle dans une nouvelle couche)
USER app

# Télécharge le modèle d'embeddings PENDANT la construction : le conteneur démarre
# vite et sans accès à Hugging Face (le modèle est inclus dans l'image)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

# Code de l'application (le notebook et l'évaluation ne sont pas nécessaires)
COPY --chown=app:app normalize.py rag.py api.py ./

# Données : événements bruts et index FAISS déjà construit.
# `app` en est propriétaire : POST /reindex peut réécrire l'index.
COPY --chown=app:app data ./data

# Le modèle est déjà dans l'image : interdit tout téléchargement au démarrage
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000

# Docker vérifie régulièrement que l'API répond (statut visible avec `docker ps`).
# start-period : le chargement du modèle prend quelques secondes au démarrage.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# --host 0.0.0.0 : écoute sur toutes les interfaces (obligatoire dans un conteneur)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

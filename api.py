"""API FastAPI du RAG. Lancement : uv run uvicorn api:app --reload  (Swagger : /docs)

Ce fichier ne contient que la couche HTTP : routes, schémas des requêtes/réponses et
traduction des erreurs en codes HTTP. Toute la logique RAG est dans `rag.py`.
FastAPI génère automatiquement la documentation Swagger (/docs) et ReDoc (/redoc)
à partir des routes, des docstrings et des schémas Pydantic ci-dessous.
"""

import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from rag import MissingApiKey, RagEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Code exécuté au démarrage (avant `yield`) et à l'arrêt (après) du serveur."""
    app.state.engine = RagEngine()  # charge le modèle d'embeddings et l'index une seule fois
    yield  # le serveur traite les requêtes pendant ce temps


# Application FastAPI : le titre, la description et la version s'affichent en haut de Swagger
app = FastAPI(
    title="API RAG — Événements OpenAgenda",
    description=(
        "Recherche sémantique et questions-réponses sur des événements OpenAgenda (FAISS + embeddings locaux + OpenAI)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def get_engine(request: Request) -> RagEngine:
    """Récupère le moteur RAG partagé, créé au démarrage."""
    return request.app.state.engine


# Injection de dépendance : une route qui déclare un paramètre `engine: Engine`
# reçoit automatiquement le moteur RAG (FastAPI appelle get_engine pour nous).
Engine = Annotated[RagEngine, Depends(get_engine)]


# ------------------------------------------------------------------ schémas
# Les schémas Pydantic valident les données entrantes (erreur 422 si invalides)
# et décrivent les réponses. Les `description` apparaissent dans Swagger.


class EventHit(BaseModel):
    """Un événement renvoyé par l'API (issu des métadonnées du document)."""

    event_id: str
    title: str
    start: str | None = Field(None, description="Date de début (ISO 8601)")
    end: str | None = Field(None, description="Date de fin (ISO 8601)")
    city: str | None = None
    venue: str | None = None
    region: str | None = None
    price_status: str = Field(description="`free`, `paid` ou `unknown` (inconnu ≠ gratuit)")
    price_details: str | None = None
    source: str | None = Field(None, description="Lien vers l'événement sur OpenAgenda")
    score: float = Field(description="Distance L2 (plus petit = plus proche)")


class AskRequest(BaseModel):
    """Corps JSON de POST /ask : seule la question est obligatoire."""

    question: str = Field(min_length=3, description="Question posée au LLM, en langage naturel")
    k: int = Field(3, ge=1, le=10, description="Nombre d'événements fournis au LLM")  # entre 1 et 10
    city: str | None = Field(None, description="Filtre facultatif : ville")
    region: str | None = Field(None, description="Filtre facultatif : région")
    department: str | None = Field(None, description="Filtre facultatif : département")
    # `pattern` : seules les trois valeurs autorisées sont acceptées (sinon erreur 422)
    price_status: str | None = Field(
        None, pattern="^(free|paid|unknown)$", description="Filtre facultatif : `free`, `paid` ou `unknown`"
    )


# Exemples proposés dans Swagger (menu déroulant de « Try it out »). Sans eux, Swagger
# préremplit les filtres avec "string", ce qui ne correspond à aucun événement.
ASK_EXAMPLES = {
    "question": {
        "summary": "Question simple",
        "value": {"question": "Y a-t-il des événements sur l'intelligence artificielle ?"},
    },
    "ville": {
        "summary": "Avec un filtre sur la ville",
        "value": {"question": "Quels événements sont prévus ?", "city": "Paris", "k": 3},
    },
    "gratuit": {
        "summary": "Seulement les événements gratuits",
        "value": {"question": "Que puis-je faire gratuitement ?", "price_status": "free"},
    },
}


class AskResponse(BaseModel):
    """Réponse de POST /ask : le texte du LLM et les événements utilisés pour le produire."""

    answer: str
    sources: list[EventHit]


class ReindexResponse(BaseModel):
    """Bilan de la réindexation."""

    documents: int  # nombre de documents (morceaux) indexés
    events: int  # nombre d'événements distincts
    rejected: list[str]  # événements rejetés à la normalisation, avec la raison


class HealthResponse(BaseModel):
    """État du service."""

    status: str  # "ok" ou "index vide"
    indexed_chunks: int  # nombre de morceaux dans l'index FAISS
    llm_configured: bool  # True si OPENAI_API_KEY est renseignée


# Champs de AskRequest qui sont des filtres sur les métadonnées (et non la question ou k)
FILTER_FIELDS = ("city", "region", "department", "price_status")


def _to_hit(doc, score: float) -> EventHit:
    """Convertit un Document LangChain en EventHit (seulement les métadonnées utiles)."""
    m = doc.metadata
    # .get() pour les champs facultatifs : une métadonnée absente signifie « inconnu »
    return EventHit(
        event_id=m["event_id"],
        title=m["title"],
        start=m.get("start"),
        end=m.get("end"),
        city=m.get("city"),
        venue=m.get("venue"),
        region=m.get("region"),
        price_status=m["price_status"],
        price_details=m.get("price_details"),
        source=m.get("source"),
        score=score,
    )


# ---------------------------------------------------------------- endpoints
# `tags` regroupe les routes par section dans Swagger ; `response_model` valide
# et documente la réponse. Les routes sont des `def` (et non `async def`) : FastAPI
# les exécute dans un thread, ce qui convient aux appels bloquants (FAISS, OpenAI).


@app.get("/", include_in_schema=False)  # include_in_schema=False : absente de Swagger
def root():
    """La racine redirige vers Swagger."""
    return RedirectResponse("/docs")


@app.get("/health", response_model=HealthResponse, tags=["Système"])
def health(engine: Engine):
    """État du service : taille de l'index et présence de la clé OpenAI."""
    return HealthResponse(
        status="ok" if engine.indexed_chunks else "index vide",
        indexed_chunks=engine.indexed_chunks,
        llm_configured=bool(os.getenv("OPENAI_API_KEY")),
    )


@app.get("/search", response_model=list[EventHit], tags=["Recherche"])
def search(
    engine: Engine,
    # Paramètres d'URL (?q=...&k=...) : Query ajoute validation et description dans Swagger
    q: Annotated[str, Query(min_length=2, description="Requête en langage naturel")],
    k: Annotated[int, Query(ge=1, le=20)] = 5,
    city: str | None = None,
    region: str | None = None,
    department: str | None = None,
    price_status: Annotated[str | None, Query(pattern="^(free|paid|unknown)$")] = None,
):
    """Recherche sémantique (sans LLM) : un résultat par événement, avec filtres sur les métadonnées."""
    # On ne transmet que les filtres réellement renseignés
    given = {"city": city, "region": region, "department": department, "price_status": price_status}
    filters = {key: value for key, value in given.items() if value is not None}
    try:
        hits = engine.search(q, k=k, **filters)
    except LookupError as exc:  # index absent
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return [_to_hit(doc, score) for doc, score in hits]


@app.post("/ask", response_model=AskResponse, tags=["Recherche"])
def ask(body: Annotated[AskRequest, Body(openapi_examples=ASK_EXAMPLES)], engine: Engine):
    """Pose une question au LLM : il répond en français à partir des événements les plus proches (RAG).

    Dans Swagger : **Try it out**, choisir un exemple ou écrire sa question, puis **Execute**.
    Seul `question` est obligatoire ; les filtres sont facultatifs.
    """
    # Filtres renseignés dans le corps de la requête (les champs à None sont ignorés)
    filters = {f: getattr(body, f) for f in FILTER_FIELDS if getattr(body, f) is not None}
    try:
        hits = engine.search(body.question, k=body.k, **filters)  # 1. retrouver les événements
        text = engine.generate(body.question, hits)  # 2. faire rédiger la réponse
    except (LookupError, MissingApiKey) as exc:
        # index absent ou clé OpenAI manquante : service indisponible
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # erreurs de l'API OpenAI (quota, clé invalide...)
        status = getattr(exc, "status_code", None)  # code HTTP renvoyé par OpenAI, s'il existe
        raise HTTPException(
            status_code=429 if status == 429 else 502,  # 429 : quota dépassé ; 502 : autre erreur du fournisseur
            detail=f"Erreur du fournisseur LLM : {getattr(exc, 'message', None) or exc}",
        ) from exc  # `from exc` : l'erreur d'origine reste visible dans les logs
    return AskResponse(answer=text, sources=[_to_hit(doc, score) for doc, score in hits])


@app.post("/reindex", response_model=ReindexResponse, tags=["Système"])
def reindex(engine: Engine):
    """Reconstruit l'index FAISS à partir de `data/events.json` (quelques secondes)."""
    try:
        documents, events, errors = engine.reindex()
    except (ValueError, FileNotFoundError) as exc:  # aucun document ou fichier absent
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ReindexResponse(documents=documents, events=events, rejected=errors)

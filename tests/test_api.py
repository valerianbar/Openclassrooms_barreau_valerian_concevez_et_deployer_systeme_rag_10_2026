"""Tests de api.py : routes, validation des entrées, traduction des erreurs en codes HTTP."""

import pytest
from conftest import FakeLLM
from fastapi.testclient import TestClient

from api import app, get_engine


@pytest.fixture
def client(engine, monkeypatch):
    """Client HTTP de test branché sur le moteur de test (faux embeddings, faux LLM).

    Le client est créé sans `with` : le démarrage de l'API (qui chargerait le vrai
    modèle d'embeddings) n'est pas exécuté ; la dépendance get_engine est remplacée.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    app.dependency_overrides[get_engine] = lambda: engine
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_racine_redirige_vers_swagger(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_swagger_et_schema_openapi(client):
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert set(schema["paths"]) == {"/health", "/search", "/ask", "/reindex"}  # « / » est masquée
    examples = schema["paths"]["/ask"]["post"]["requestBody"]["content"]["application/json"]["examples"]
    assert set(examples) == {"question", "ville", "gratuit"}


def test_health(client, engine):
    body = client.get("/health").json()
    assert body == {"status": "ok", "indexed_chunks": engine.indexed_chunks, "llm_configured": True}


def test_health_sans_cle(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    assert client.get("/health").json()["llm_configured"] is False


# ---------------------------------------------------------------------- /search


def test_search(client):
    response = client.get("/search", params={"q": "concert", "k": 2})
    assert response.status_code == 200
    hits = response.json()
    assert len(hits) == 2
    assert {"event_id", "title", "city", "price_status", "source", "score"} <= set(hits[0])


def test_search_avec_filtre(client):
    hits = client.get("/search", params={"q": "événement", "city": "Caen"}).json()
    assert [h["event_id"] for h in hits] == ["2"]


@pytest.mark.parametrize(
    "params",
    [
        {"q": "x"},  # requête trop courte
        {"q": "concert", "k": 0},  # k hors limites
        {"q": "concert", "price_status": "gratuit"},  # valeur non autorisée
        {},  # q obligatoire
    ],
)
def test_search_entree_invalide(client, params):
    assert client.get("/search", params=params).status_code == 422


def test_search_sans_index(client, engine):
    engine.vectorstore = None
    response = client.get("/search", params={"q": "concert"})
    assert response.status_code == 503


# ------------------------------------------------------------------------- /ask


def test_ask(client, engine):
    response = client.post("/ask", json={"question": "Quels concerts à Paris ?", "k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Réponse simulée."
    assert len(body["sources"]) == 2
    assert len(engine._llm.calls) == 1


def test_ask_transmet_les_filtres(client):
    body = client.post("/ask", json={"question": "Que faire ?", "city": "Brest"}).json()
    assert [s["event_id"] for s in body["sources"]] == ["3"]


def test_ask_question_obligatoire(client):
    assert client.post("/ask", json={"k": 3}).status_code == 422


def test_ask_sans_cle_openai(client, engine, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    engine._llm = None
    response = client.post("/ask", json={"question": "Quels concerts ?"})
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


class QuotaError(Exception):
    """Imite une erreur de l'API OpenAI (qui porte un attribut status_code)."""

    def __init__(self, status_code):
        super().__init__("erreur simulée")
        self.status_code = status_code


@pytest.mark.parametrize("status_code, expected", [(429, 429), (401, 502), (500, 502)])
def test_ask_erreur_du_fournisseur_llm(client, engine, status_code, expected):
    engine._llm = FakeLLM(error=QuotaError(status_code))
    response = client.post("/ask", json={"question": "Quels concerts ?"})
    assert response.status_code == expected
    assert response.json()["detail"].startswith("Erreur du fournisseur LLM")


# --------------------------------------------------------------------- /reindex


def test_reindex(client):
    response = client.post("/reindex")
    assert response.status_code == 200
    assert response.json()["events"] == 4

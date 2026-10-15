"""Données et objets partagés par les tests.

Les tests unitaires ne téléchargent aucun modèle et n'appellent pas OpenAI :
    - les embeddings sont remplacés par `DeterministicFakeEmbedding` (vecteurs calculés
      à partir du texte : un même texte donne toujours le même vecteur) ;
    - le LLM est remplacé par `FakeLLM`, qui renvoie une réponse fixe et garde le prompt reçu.
"""

import json

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage

from rag import RagEngine


def make_raw_event(uid="1", **overrides) -> dict:
    """Événement brut au format de l'API OpenAgenda (OpenDataSoft), modifiable champ par champ."""
    raw = {
        "uid": uid,
        "title_fr": f"Événement {uid}",
        "description_fr": "Courte description.",
        "longdescription_fr": "<p>Longue description de l'événement.</p>",
        "firstdate_begin": "2026-10-08T07:00:00+00:00",
        "lastdate_end": "2026-10-08T15:00:00+00:00",
        "location_name": "Salle des fêtes",
        "location_address": "1 rue de la Paix",
        "location_city": "Paris",
        "location_postalcode": "75001",
        "location_department": "Paris",
        "location_region": "Île-de-France",
        "location_coordinates": {"lat": 48.86, "lon": 2.33},
        "keywords_fr": ["concert", "jazz"],
        "conditions_fr": None,
        "canonicalurl": f"https://openagenda.com/test/events/{uid}",
        "category": None,
        # données personnelles présentes dans la source : ne doivent jamais être reprises
        "contributor_email": "contact@example.com",
        "contributor_contactnumber": "0102030405",
    }
    raw.update(overrides)
    return raw


class FakeLLM:
    """Remplace ChatOpenAI : renvoie `answer` et mémorise les messages reçus."""

    def __init__(self, answer="Réponse simulée.", error: Exception | None = None):
        self.answer = answer
        self.error = error
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return AIMessage(content=self.answer)


@pytest.fixture
def raw_event():
    """Fabrique d'événements bruts : raw_event("42", location_city="Lyon")."""
    return make_raw_event


@pytest.fixture
def sample_events():
    """Petit jeu d'événements : trois villes, un tarif de chaque type, une description longue."""
    return [
        make_raw_event("1", title_fr="Concert de jazz", location_city="Paris", conditions_fr="Gratuit"),
        make_raw_event("2", title_fr="Festival electro", location_city="Caen", conditions_fr="De 6 € à 60 €"),
        make_raw_event("3", title_fr="Atelier poterie", location_city="Brest", conditions_fr=None),
        make_raw_event(
            "4",
            title_fr="Résidence d'architecte",
            location_city="Salins-les-Bains",
            # description longue : sera découpée en plusieurs morceaux
            longdescription_fr="<p>" + "Une phrase sur la résidence d'architecture. " * 80 + "</p>",
        ),
    ]


@pytest.fixture
def engine(tmp_path, sample_events):
    """Moteur RAG indexé sur `sample_events`, avec faux embeddings et faux LLM."""
    events_file = tmp_path / "events.json"
    events_file.write_text(json.dumps(sample_events), encoding="utf-8")
    rag_engine = RagEngine(
        embeddings=DeterministicFakeEmbedding(size=64),
        index_dir=tmp_path / "faiss_index",
        events_file=events_file,
    )
    rag_engine.reindex()
    rag_engine._llm = FakeLLM()  # aucun appel réseau
    return rag_engine

"""Tests de rag.py : index, recherche regroupée par événement, filtres, génération."""

import json

import pytest
from conftest import FakeLLM, make_raw_event
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag import MissingApiKey, RagEngine

# ---------------------------------------------------------------- indexation


def test_reindex_compte_documents_et_evenements(engine):
    documents, events, errors = engine.reindex()
    assert events == 4
    assert documents > events  # la description longue de l'événement 4 est découpée
    assert errors == []
    assert engine.indexed_chunks == documents


def test_index_sauvegarde_puis_recharge(engine):
    reloaded = RagEngine(embeddings=engine.embeddings, index_dir=engine.index_dir, events_file=engine.events_file)
    assert reloaded.indexed_chunks == engine.indexed_chunks


def test_sans_index_la_recherche_echoue(tmp_path):
    empty = RagEngine(embeddings=DeterministicFakeEmbedding(size=8), index_dir=tmp_path / "absent")
    assert empty.indexed_chunks == 0
    with pytest.raises(LookupError):
        empty.search("concert")


def test_reindex_sans_document_echoue(tmp_path):
    events_file = tmp_path / "events.json"
    events_file.write_text("[]", encoding="utf-8")
    rag_engine = RagEngine(
        embeddings=DeterministicFakeEmbedding(size=8), index_dir=tmp_path / "i", events_file=events_file
    )
    with pytest.raises(ValueError):
        rag_engine.reindex()


# ----------------------------------------------------------------- recherche


def test_recherche_un_resultat_par_evenement(engine):
    # requête identique au texte de l'événement découpé : tous ses morceaux sont proches
    query = engine.vectorstore.similarity_search("Résidence d'architecte", k=1)[0].page_content
    hits = engine.search(query, k=4)
    event_ids = [doc.metadata["event_id"] for doc, _ in hits]
    assert len(event_ids) == len(set(event_ids)) == 4


def test_recherche_respecte_k(engine):
    assert len(engine.search("concert", k=2)) == 2


def test_recherche_renvoie_des_scores_tries(engine):
    scores = [score for _, score in engine.search("concert", k=4)]
    assert all(isinstance(s, float) for s in scores)
    assert scores == sorted(scores)  # distance : du plus proche au plus lointain


@pytest.mark.parametrize(
    "filters, expected",
    [
        ({"city": "Caen"}, {"2"}),
        ({"price_status": "free"}, {"1"}),
        ({"price_status": "unknown"}, {"3", "4"}),
    ],
)
def test_recherche_avec_filtres(engine, filters, expected):
    hits = engine.search("événement", k=5, **filters)
    assert {doc.metadata["event_id"] for doc, _ in hits} == expected


def test_filtre_trouve_un_evenement_peu_proche(tmp_path):
    """Avec un filtre, tout l'index est examiné (et pas seulement les 20 plus proches)."""
    events = [make_raw_event(str(i), location_city="Paris") for i in range(40)]
    events.append(make_raw_event("brest", title_fr="Fest-noz", location_city="Brest"))
    events_file = tmp_path / "events.json"
    events_file.write_text(json.dumps(events), encoding="utf-8")
    rag_engine = RagEngine(
        embeddings=DeterministicFakeEmbedding(size=64), index_dir=tmp_path / "i", events_file=events_file
    )
    rag_engine.reindex()
    hits = rag_engine.search("Événement 1", k=3, city="Brest")
    assert [doc.metadata["event_id"] for doc, _ in hits] == ["brest"]


# ---------------------------------------------------------------- génération


def test_generate_sans_evenement_n_appelle_pas_le_llm(engine):
    answer = engine.generate("question", [])
    assert answer == "Je n'ai trouvé aucun événement correspondant."
    assert engine._llm.calls == []


def test_generate_envoie_contexte_question_et_date(engine):
    hits = engine.search("concert", k=2)
    answer = engine.generate("Quels concerts ?", hits)
    assert answer == "Réponse simulée."
    system, human = engine._llm.calls[0].to_messages()
    assert "UNIQUEMENT" in system.content  # consigne anti-hallucination
    assert "Date du jour" in system.content
    assert "Question : Quels concerts ?" in human.content
    for doc, _ in hits:
        assert doc.page_content in human.content


def test_cle_openai_absente(engine, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    engine._llm = None
    with pytest.raises(MissingApiKey):
        engine.generate("question", engine.search("concert", k=1))


def test_llm_cree_avec_le_modele_du_env(engine, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    engine._llm = None
    llm = engine._get_llm()  # aucun appel réseau : le client est seulement créé
    assert llm.model_name == "gpt-test"
    assert engine._get_llm() is llm  # créé une seule fois


def test_fake_llm_est_bien_utilise(engine):
    assert isinstance(engine._llm, FakeLLM)

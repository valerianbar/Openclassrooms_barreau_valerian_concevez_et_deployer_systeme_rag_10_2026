"""Tests de normalize.py : nettoyage, tarif, schéma Pydantic, conversion en Documents."""

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from normalize import (
    PriceStatus,
    clean_html,
    clean_text,
    event_to_documents,
    load_documents,
    normalize_event,
    parse_price,
)

# ------------------------------------------------------------------ nettoyage


def test_clean_html_retire_les_balises_et_garde_les_paragraphes():
    text = clean_html("<p>Premier   paragraphe.</p><p>Second<br>ligne</p>")
    assert "<" not in text
    # chaque paragraphe et chaque ligne (<br>) reste sur sa propre ligne, espaces normalisés
    assert [line for line in text.splitlines() if line] == ["Premier paragraphe.", "Second", "ligne"]
    assert "\n\n\n" not in text  # au maximum une ligne vide


def test_clean_html_transforme_les_listes_en_puces():
    text = clean_html("<ul><li>Un</li><li>Deux</li></ul>")
    assert "- Un" in text and "- Deux" in text


def test_clean_html_remplace_les_espaces_insecables():
    assert clean_html("Entrée\xa0libre") == "Entrée libre"


@pytest.mark.parametrize("raw", [None, "", "<p>   </p>"])
def test_clean_html_renvoie_none_si_vide(raw):
    assert clean_html(raw) is None


def test_clean_text_met_tout_sur_une_ligne():
    assert clean_text("<p>Salle\n   des fêtes</p>") == "Salle des fêtes"


# ---------------------------------------------------------------------- tarif


@pytest.mark.parametrize(
    "conditions, expected",
    [
        ("Gratuit", PriceStatus.FREE),
        ("gratuit.", PriceStatus.FREE),
        ("Entrée libre", PriceStatus.FREE),
        ("Entrée libre et gratuite", PriceStatus.FREE),
        ("Tarifs : ADULTE 390€", PriceStatus.PAID),
        ("CHF 500.– tarif standard", PriceStatus.PAID),
        ("12 euros", PriceStatus.PAID),
        # « gratuit » dans une phrase ne suffit pas : l'entrée elle-même n'est pas dite gratuite
        ("Inscription gratuite mais obligatoire", PriceStatus.UNKNOWN),
        ("Sur inscription", PriceStatus.UNKNOWN),
    ],
)
def test_parse_price(conditions, expected):
    assert parse_price(conditions).status is expected


@pytest.mark.parametrize("conditions", [None, ""])
def test_prix_absent_n_est_pas_gratuit(conditions):
    price = parse_price(conditions)
    assert price.status is PriceStatus.UNKNOWN
    assert price.details is None


def test_parse_price_conserve_le_texte_d_origine():
    assert parse_price("<p>Sur inscription</p>").details == "Sur inscription"


# -------------------------------------------------------------- normalisation


def test_normalize_event_remplit_tous_les_champs(raw_event):
    event = normalize_event(raw_event("42", conditions_fr="Gratuit"))
    assert event.id == "42"
    assert event.title == "Événement 42"
    assert event.city == "Paris"
    assert event.region == "Île-de-France"
    assert (event.latitude, event.longitude) == (48.86, 2.33)
    assert event.keywords == ["concert", "jazz"]
    assert event.price.status is PriceStatus.FREE
    assert event.url == "https://openagenda.com/test/events/42"


def test_normalize_event_convertit_les_dates(raw_event):
    event = normalize_event(raw_event())
    assert event.start == datetime(2026, 10, 8, 7, 0, tzinfo=timezone.utc)
    assert event.end == datetime(2026, 10, 8, 15, 0, tzinfo=timezone.utc)


def test_normalize_event_combine_description_courte_et_longue(raw_event):
    event = normalize_event(raw_event())
    assert event.description == "Courte description.\n\nLongue description de l'événement."


def test_normalize_event_evite_les_doublons_de_description(raw_event):
    event = normalize_event(raw_event(description_fr="Longue description"))
    assert event.description == "Longue description de l'événement."


def test_normalize_event_champs_absents_restent_inconnus(raw_event):
    event = normalize_event(raw_event(location_city=None, location_coordinates=None, keywords_fr=None))
    assert event.city is None
    assert event.latitude is None
    assert event.keywords == []


@pytest.mark.parametrize("title", ["", None, "<p> </p>"])
def test_normalize_event_rejette_un_titre_vide(raw_event, title):
    with pytest.raises(ValidationError, match="titre vide"):
        normalize_event(raw_event(title_fr=title))


# ----------------------------------------------------------------- Documents


def test_un_document_par_evenement_court(raw_event):
    docs = event_to_documents(normalize_event(raw_event("7")))
    assert len(docs) == 1
    assert docs[0].metadata["event_id"] == "7"
    assert docs[0].metadata["chunk_count"] == 1
    assert docs[0].page_content.startswith("Titre : Événement 7")


def test_le_texte_indique_tarif_non_precise_si_inconnu(raw_event):
    doc = event_to_documents(normalize_event(raw_event(conditions_fr=None)))[0]
    assert "Tarif : Non précisé" in doc.page_content
    assert "Gratuit" not in doc.page_content


def test_metadonnees_plates_et_sans_valeur_inconnue(raw_event):
    meta = event_to_documents(normalize_event(raw_event(location_region=None)))[0].metadata
    # uniquement des scalaires : compatible avec les filtres des bases vectorielles
    assert all(isinstance(v, (str, int, float)) for v in meta.values())
    assert "region" not in meta  # inconnu = clé absente (et non None)
    assert meta["keywords"] == "concert, jazz"
    assert meta["start_ts"] == int(datetime(2026, 10, 8, 7, tzinfo=timezone.utc).timestamp())
    assert meta["price_status"] == "unknown"


def test_lieu_sans_repetition(raw_event):
    raw = raw_event(location_name="Ouessant, France", location_address=None, location_city="Ouessant")
    doc = event_to_documents(normalize_event(raw))[0]
    assert "Lieu : Ouessant, France\n" in doc.page_content


def test_description_longue_decoupee_en_gardant_les_metadonnees(raw_event):
    long_html = "<p>" + "Une phrase assez longue pour remplir la description. " * 60 + "</p>"
    docs = event_to_documents(normalize_event(raw_event("9", longdescription_fr=long_html)))
    assert len(docs) > 1
    assert [d.metadata["chunk_index"] for d in docs] == list(range(len(docs)))
    for doc in docs:
        assert doc.metadata["event_id"] == "9"
        assert doc.metadata["chunk_count"] == len(docs)
        assert doc.page_content.startswith("Titre : Événement 9")  # en-tête répété sur chaque morceau


def test_les_coordonnees_des_contributeurs_ne_sont_pas_reprises(raw_event):
    doc = event_to_documents(normalize_event(raw_event()))[0]
    content = doc.page_content + json.dumps(doc.metadata)
    assert "contact@example.com" not in content
    assert "0102030405" not in content


def test_load_documents_ignore_les_evenements_invalides(tmp_path, raw_event):
    path = tmp_path / "events.json"
    path.write_text(json.dumps([raw_event("1"), raw_event("2", title_fr="")]), encoding="utf-8")
    documents, errors = load_documents(path)
    assert {d.metadata["event_id"] for d in documents} == {"1"}
    assert len(errors) == 1 and errors[0].startswith("2:")

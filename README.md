# Système RAG sur les événements OpenAgenda

Assistant de questions-réponses sur des événements culturels et publics issus d'[OpenAgenda](https://openagenda.com) : récupération, normalisation, indexation vectorielle (FAISS), recherche sémantique et génération de réponses avec un LLM (OpenAI).

## Fonctionnement

1. **Récupération** : 200 événements à venir, via le jeu de données public OpenAgenda hébergé sur OpenDataSoft (aucune clé requise).
2. **Normalisation** : validation avec Pydantic, nettoyage du HTML et des espaces, dates structurées.
3. **Documents LangChain** : un document par événement (texte lisible + métadonnées). Seules les descriptions de plus de 1000 caractères sont découpées ; chaque morceau conserve l'en-tête, l'identifiant et les métadonnées.
4. **Indexation** : embeddings locaux `intfloat/multilingual-e5-small`, index FAISS sauvegardé sur disque.
5. **Recherche** : similarité sémantique avec filtres sur les métadonnées, résultats regroupés par événement.
6. **Génération** : les 3 événements les plus proches sont envoyés à un LLM OpenAI (`gpt-4o-mini` par défaut) qui répond en français et cite titre, date, lieu et lien source.

### Gestion du tarif

Un prix absent ne veut pas dire « gratuit ». Le champ `price.status` prend trois valeurs :

| Valeur    | Condition                                                         |
|-----------|-------------------------------------------------------------------|
| `free`    | le texte affirme clairement « Gratuit » ou « Entrée libre »       |
| `paid`    | un montant apparaît (ex. `390€`, `CHF 500`)                       |
| `unknown` | information absente ou ambiguë (ex. « Inscription gratuite mais obligatoire ») |

Le prompt interdit au LLM de présenter comme gratuit un événement dont le tarif est « Non précisé ».

## Structure

| Fichier              | Rôle                                                              |
|----------------------|-------------------------------------------------------------------|
| `rag_events.ipynb`   | Pipeline complet (données → FAISS → recherche → génération) + client de l'API |
| `normalize.py`       | Schéma Pydantic, nettoyage, conversion en `Document`              |
| `rag.py`             | Moteur RAG : index FAISS, recherche, génération OpenAI            |
| `api.py`             | API FastAPI (Swagger sur `/docs`)                                 |
| `data/`              | Données brutes et index FAISS (générés)                           |
| `.env`               | Configuration et clés (non versionné)                             |

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/) et Python 3.13.

```bash
uv sync
```

Le premier lancement télécharge le modèle d'embeddings (environ 500 Mo avec PyTorch).

## Configuration

Dans `.env` (sans espace autour du `=`, sans guillemets) :

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

La clé se crée sur [platform.openai.com/api-keys](https://platform.openai.com/api-keys) et le compte doit disposer de crédit. Les embeddings étant locaux, ils ne nécessitent aucune clé : seule l'étape de génération appelle l'API.

## Utilisation

Ouvrir `rag_events.ipynb`, sélectionner le noyau `.venv` du projet, puis exécuter les cellules dans l'ordre.

```python
search("randonnée nature en Bretagne")
search("concert", city="Paris")  # filtre sur les métadonnées

reply, sources = answer("Y a-t-il des événements sur l'intelligence artificielle ?")
print(reply)
```

### API et Swagger

L'API se lance **depuis le terminal**, après avoir créé l'index (sections 1 à 5 du notebook, ou `POST /reindex`) :

```bash
uv run uvicorn api:app --reload
```

La section 8 du notebook contient un client qui l'interroge.

- Swagger (interface interactive) : <http://127.0.0.1:8000/docs>
- ReDoc : <http://127.0.0.1:8000/redoc>

| Méthode | Route      | Rôle                                                                 |
|---------|------------|----------------------------------------------------------------------|
| `GET`   | `/health`  | État du service (taille de l'index, clé OpenAI présente)             |
| `GET`   | `/search`  | Recherche sémantique sans LLM, filtres `city`, `region`, `department`, `price_status` |
| `POST`  | `/ask`     | Réponse en français générée par OpenAI + événements sources          |
| `POST`  | `/reindex` | Reconstruit l'index FAISS depuis `data/events.json`                  |

```bash
curl "http://127.0.0.1:8000/search?q=emploi&city=Marseille&k=3"
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "Y a-t-il des événements sur l'\''IA ?", "k": 3}'
```

Erreurs : `503` si l'index ou `OPENAI_API_KEY` manque, `429` si le quota OpenAI est dépassé, `502` pour les autres erreurs du fournisseur.


## Métadonnées des documents

`event_id`, `title`, `start`, `end`, `start_ts`, `end_ts`, `venue`, `city`, `postal_code`, `department`, `region`, `latitude`, `longitude`, `category`, `keywords`, `price_status`, `price_details`, `source`, `chunk_index`, `chunk_count`.

Les dates existent en ISO (`start`) et en timestamp (`start_ts`) pour faciliter les filtres.

## Limites connues

- La source ne renseigne ni la catégorie (0 %) ni le tarif (environ 6 % des événements) ; les mots-clés servent de substitut pour la catégorie.
- Un événement est rejeté à la normalisation car son titre est vide.
- Les 200 événements sont les premiers à venir par ordre de date : ils sont concentrés sur quelques jours et en grande partie issus de France Travail.
- Les coordonnées des contributeurs (e-mail, téléphone) présentes dans les données brutes ne sont pas reprises dans les documents.

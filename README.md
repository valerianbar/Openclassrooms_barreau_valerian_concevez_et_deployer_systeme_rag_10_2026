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
| `evaluate_ragas.py`  | Évaluation du RAG avec Ragas                                      |
| `evaluation/`        | Jeu de test Ragas et résultats                                    |
| `tests/`             | Tests unitaires (pytest)                                          |
| `.github/workflows/` | Pipeline CI/CD (GitHub Actions)                                   |
| `Dockerfile`         | Image Docker de l'API                                             |
| `.dockerignore`      | Fichiers exclus de l'image (dont `.env`)                          |
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


## Tests unitaires

```bash
uv run pytest
```

69 tests, exécutés en environ une seconde, **sans réseau** : les embeddings sont remplacés par des vecteurs déterministes et le LLM par une réponse simulée (aucun modèle téléchargé, aucun appel à OpenAI).

| Fichier                   | Ce qui est testé                                                              |
|---------------------------|-------------------------------------------------------------------------------|
| `tests/test_normalize.py` | nettoyage HTML, tarif (`free`/`paid`/`unknown`), schéma Pydantic, Documents, données personnelles exclues |
| `tests/test_rag.py`       | indexation, recherche regroupée par événement, filtres, prompt envoyé au LLM, clé absente |
| `tests/test_api.py`       | routes, validation des entrées (422), erreurs traduites en 503 / 429 / 502, Swagger |

Couverture du code : `uv run pytest --cov=normalize --cov=rag --cov=api --cov-report=term-missing` (93 %).

## CI/CD (GitHub Actions)

Le pipeline `.github/workflows/ci-cd.yml` se lance à chaque push et pull request :

1. **Tests unitaires** : installation des dépendances avec `uv sync --locked`, puis `pytest` avec la couverture.
2. **Image Docker** (si les tests passent) : construction de l'image, puis test de fumée du conteneur (`/health`, `/search`, `/docs`, et `/ask` qui doit répondre 503 sans clé OpenAI).
3. **Publication** (uniquement sur `main`) : l'image est publiée sur GitHub Container Registry, avec les tags `latest` et `sha-<commit>`.

Aucun secret n'est à configurer : les tests n'appellent pas OpenAI et la publication utilise le `GITHUB_TOKEN` fourni par GitHub. Récupérer l'image publiée :

```bash
docker pull ghcr.io/valerianbar/openclassrooms_barreau_valerian_concevez_et_deployer_systeme_rag_10_2026:latest
```

L'évaluation Ragas n'est pas dans le pipeline : elle est payante et nécessite la clé OpenAI.

## Docker

L'image contient l'API, l'index FAISS (`data/`) et le modèle d'embeddings : le conteneur démarre en quelques secondes, sans téléchargement. La clé OpenAI n'est **pas** dans l'image, elle est passée au lancement.

```bash
docker build -t rag-openagenda .
```

```bash
docker run -d --name rag -p 8000:8000 --env-file .env rag-openagenda
```

Swagger : <http://localhost:8000/docs>. Arrêt : `docker rm -f rag`.

- Image d'environ 2,9 Go (PyTorch CPU ~650 Mo, modèle d'embeddings ~470 Mo).
- Seules les dépendances de l'API sont installées : le notebook et Ragas sont dans le groupe `dev` de `pyproject.toml` (`uv sync` les installe en local, pas dans l'image).
- L'API tourne avec un utilisateur non-root ; `docker ps` affiche son état de santé (`healthy`).
- Un `POST /reindex` réécrit l'index **dans le conteneur** : il est perdu si le conteneur est supprimé. Pour le conserver, monter un volume : `-v ./data:/app/data`.

## Évaluation (Ragas)

```bash
uv run python evaluate_ragas.py
```

Le script pose les 11 questions de `evaluation/ragas_dataset.json` au RAG (comme `/ask`), puis un LLM juge (`gpt-4o-mini`, modifiable avec `RAGAS_JUDGE_MODEL`) note les réponses. Coût : quelques centimes. Le script renvoie un code d'erreur si une moyenne passe sous son seuil.

| Métrique              | Ce qu'elle mesure                                               | Seuil | Résultat |
|-----------------------|-----------------------------------------------------------------|-------|----------|
| `faithfulness`        | la réponse ne contient que des faits présents dans le contexte  | 0,8   | 0,97     |
| `answer_relevancy`    | la réponse répond à la question posée                           | 0,7   | 0,66     |
| `context_precision`   | les contextes utiles sont classés en tête                       | 0,7   | 1,00     |
| `context_recall`      | les contextes contiennent les infos de la réponse attendue      | 0,7   | 0,92     |
| `factual_correctness` | la réponse est d'accord avec la réponse attendue                | 0,5   | 0,51     |
| `retrieval_hit`       | l'événement attendu est retrouvé (sans LLM)                     | 0,8   | 1,00     |

Résultats détaillés (réponses, contextes, scores) : `evaluation/ragas_results.json` et `.csv`.

`answer_relevancy` est sous son seuil pour deux raisons : certaines réponses recopient les champs (titre, lieu, tarif...) au lieu de répondre d'abord à la question, et Ragas note 0 une réponse prudente comme « tarif non précisé », alors que c'est le comportement voulu.

## Métadonnées des documents

`event_id`, `title`, `start`, `end`, `start_ts`, `end_ts`, `venue`, `city`, `postal_code`, `department`, `region`, `latitude`, `longitude`, `category`, `keywords`, `price_status`, `price_details`, `source`, `chunk_index`, `chunk_count`.

Les dates existent en ISO (`start`) et en timestamp (`start_ts`) pour faciliter les filtres.

## Limites connues

- La source ne renseigne ni la catégorie (0 %) ni le tarif (environ 6 % des événements) ; les mots-clés servent de substitut pour la catégorie.
- Un événement est rejeté à la normalisation car son titre est vide.
- Les 200 événements sont les premiers à venir par ordre de date : ils sont concentrés sur quelques jours et en grande partie issus de France Travail.
- Les coordonnées des contributeurs (e-mail, téléphone) présentes dans les données brutes ne sont pas reprises dans les documents.

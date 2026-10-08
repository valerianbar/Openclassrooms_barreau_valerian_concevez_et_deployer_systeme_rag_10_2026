"""Cœur du RAG : index FAISS, recherche regroupée par événement, génération avec OpenAI.

Ce module ne connaît rien de HTTP : il est utilisé par `api.py`, qui l'expose en API REST.
Étapes d'une question :
    1. la question est transformée en vecteur (embedding E5) ;
    2. FAISS retrouve les morceaux d'événements les plus proches ;
    3. on ne garde qu'un morceau par événement ;
    4. ces événements servent de contexte au LLM OpenAI, qui rédige la réponse.
"""

import os
from datetime import date
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from normalize import load_documents  # données brutes -> Documents LangChain (voir normalize.py)

# Charge les variables du fichier .env (OPENAI_API_KEY, OPENAI_MODEL).
# usecwd=True : cherche le .env à partir du dossier où l'on lance la commande (uvicorn).
# override=True : les valeurs du .env remplacent celles déjà présentes dans l'environnement.
load_dotenv(find_dotenv(usecwd=True), override=True)

# --- Emplacements des données (chemins relatifs à la racine du projet) ---
INDEX_DIR = Path("data/faiss_index")   # index FAISS sauvegardé (créé par le notebook ou /reindex)
EVENTS_FILE = Path("data/events.json")  # événements bruts téléchargés depuis OpenAgenda

# --- Consignes données au LLM ---
# Le message « system » fixe les règles ; le message « human » contient le contexte et la question.
# Les règles limitent les hallucinations : répondre UNIQUEMENT à partir du contexte,
# citer les sources, et ne jamais transformer un tarif inconnu en « gratuit ».
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un assistant qui renseigne sur des événements culturels et publics.\n"
     "Réponds en français, de façon naturelle et concise, en t'appuyant UNIQUEMENT sur les "
     "événements du contexte. Si le contexte ne permet pas de répondre, dis-le clairement.\n"
     "Pour chaque événement cité, indique le titre, la date, le lieu et le lien source.\n"
     "Si le tarif est « Non précisé », ne dis jamais que l'événement est gratuit.\n"
     "Date du jour : {today}."),  # la date permet d'interpréter « demain », « ce week-end »...
    ("human", "Contexte :\n{context}\n\nQuestion : {question}"),
])


class E5Embeddings(HuggingFaceEmbeddings):
    """E5 attend un préfixe différent pour les documents et les requêtes.

    Sans ces préfixes, le modèle compare moins bien questions et documents.
    """

    def embed_documents(self, texts):
        # textes indexés -> préfixe « passage: »
        return super().embed_documents([f"passage: {t}" for t in texts])

    def embed_query(self, text):
        # question de l'utilisateur -> préfixe « query: »
        return super().embed_query(f"query: {text}")


class MissingApiKey(RuntimeError):
    """Levée quand OPENAI_API_KEY est absente : l'API la traduit en erreur HTTP 503."""


class RagEngine:
    """Regroupe le modèle d'embeddings, l'index FAISS et le LLM.

    Une seule instance est créée au démarrage de l'API (voir `lifespan` dans api.py),
    car charger le modèle d'embeddings prend plusieurs secondes.
    """

    def __init__(self) -> None:
        # Modèle d'embeddings local et multilingue : gratuit, sans clé, bon en français.
        # Il est téléchargé au premier lancement puis mis en cache sur le disque.
        self.embeddings = E5Embeddings(
            model_name="intfloat/multilingual-e5-small",
            encode_kwargs={"normalize_embeddings": True},  # vecteurs de norme 1 : distances comparables
        )
        self.vectorstore: FAISS | None = None  # None tant qu'aucun index n'est disponible
        self._llm = None  # LLM créé seulement au premier besoin (voir _get_llm)
        if INDEX_DIR.exists():
            # Recharge l'index sauvegardé. allow_dangerous_deserialization est nécessaire
            # car l'index contient un fichier pickle : acceptable ici, car c'est un fichier
            # que l'on a produit nous-mêmes (ne jamais charger un index venant d'ailleurs).
            self.vectorstore = FAISS.load_local(
                str(INDEX_DIR), self.embeddings, allow_dangerous_deserialization=True
            )

    @property
    def indexed_chunks(self) -> int:
        """Nombre de morceaux (chunks) dans l'index, 0 s'il n'y a pas d'index."""
        return self.vectorstore.index.ntotal if self.vectorstore else 0

    def reindex(self) -> tuple[int, int, list[str]]:
        """Reconstruit l'index à partir de data/events.json. Renvoie (documents, événements, erreurs)."""
        # Normalisation + conversion en Documents ; les événements invalides sont listés dans `errors`
        documents, errors = load_documents(EVENTS_FILE)
        if not documents:
            raise ValueError("Aucun document à indexer")
        # Calcule les embeddings de tous les documents et construit un nouvel index
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        # Sauvegarde sur disque : il sera rechargé au prochain démarrage de l'API
        self.vectorstore.save_local(str(INDEX_DIR))
        # nombre d'événements distincts (un événement peut produire plusieurs documents)
        return len(documents), len({d.metadata["event_id"] for d in documents}), errors

    def search(self, query: str, k: int = 3, fetch_k: int = 20, **filters) -> list[tuple[Document, float]]:
        """Recherche sémantique ; ne garde que le meilleur morceau de chaque événement.

        `filters` filtre sur les métadonnées (ex. city="Paris", price_status="free").
        Renvoie des couples (document, score), le score étant une distance : plus petit = plus proche.
        """
        if self.vectorstore is None:
            # LookupError est traduite en HTTP 503 par l'API
            raise LookupError("Index absent : lancer POST /reindex")
        # FAISS filtre APRÈS la recherche : avec un filtre, on examine tout l'index
        # pour ne pas rater les événements qui correspondent mais sont moins proches.
        if filters:
            fetch_k = self.indexed_chunks
        # On demande plus de résultats que `k` (fetch_k) car plusieurs morceaux
        # d'un même événement peuvent arriver en tête : on les dédoublonne ensuite.
        results = self.vectorstore.similarity_search_with_score(
            query, k=max(fetch_k, k), fetch_k=max(fetch_k, k), filter=filters or None
        )
        seen: set[str] = set()  # identifiants d'événements déjà retenus
        unique: list[tuple[Document, float]] = []
        for doc, score in results:  # les résultats arrivent du plus proche au plus lointain
            if doc.metadata["event_id"] not in seen:
                seen.add(doc.metadata["event_id"])
                unique.append((doc, float(score)))  # float() : le score FAISS est un type numpy
            if len(unique) == k:  # on a assez d'événements distincts
                break
        return unique

    def _get_llm(self):
        """Crée le client OpenAI au premier appel, puis le réutilise.

        Création « paresseuse » : l'API peut démarrer (et /search fonctionner) même sans clé OpenAI.
        """
        if self._llm is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise MissingApiKey("OPENAI_API_KEY est vide ou absente du .env")
            # import local : langchain_openai n'est chargé que si l'on génère vraiment une réponse
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),  # modèle choisi dans .env
                api_key=api_key,
                temperature=0.2,  # faible : réponses fidèles au contexte, peu d'invention
            )
        return self._llm

    def generate(self, question: str, hits: list[tuple[Document, float]]) -> str:
        """Demande au LLM de répondre à partir des événements trouvés."""
        if not hits:
            # Pas d'événement : inutile (et coûteux) d'appeler le LLM
            return "Je n'ai trouvé aucun événement correspondant."
        # Le contexte = textes des événements (titre, date, lieu, tarif, description), séparés par un trait
        context = "\n\n---\n\n".join(doc.page_content for doc, _ in hits)
        # Remplit le prompt avec la date du jour, le contexte et la question
        messages = PROMPT.invoke(
            {"today": date.today().strftime("%d/%m/%Y"), "context": context, "question": question}
        )
        # Appel à l'API OpenAI (payant) ; .content = texte de la réponse
        return self._get_llm().invoke(messages).content

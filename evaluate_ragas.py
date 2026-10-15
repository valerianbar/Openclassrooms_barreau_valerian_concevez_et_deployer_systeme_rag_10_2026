"""Évaluation du RAG avec Ragas.

Lancement : uv run python evaluate_ragas.py

Pour chaque question du jeu de test (evaluation/ragas_dataset.json) :
    1. le RAG (rag.py) retrouve les événements et génère une réponse, comme le ferait /ask ;
    2. Ragas note la réponse et les contextes avec un LLM « juge » (OpenAI).

Métriques (entre 0 et 1, plus haut = mieux) :
    - faithfulness        : la réponse ne contient que des faits présents dans les contextes (anti-hallucination)
    - answer_relevancy    : la réponse répond bien à la question posée
    - context_precision   : les contextes utiles à la réponse attendue sont bien classés en tête
    - context_recall      : les contextes contiennent toutes les infos de la réponse attendue
    - factual_correctness : la réponse est factuellement d'accord avec la réponse attendue
    - retrieval_hit       : (sans LLM) l'événement attendu fait partie des événements retrouvés

Le script échoue (code de sortie 1) si une moyenne passe sous son seuil : il peut servir de test.
"""

import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path

# Ragas envoie par défaut des statistiques d'usage anonymes : on les désactive.
# (doit être défini avant l'import de ragas)
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

import pandas as pd
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)

from rag import RagEngine  # charge aussi le .env (OPENAI_API_KEY)

DATASET = Path("evaluation/ragas_dataset.json")
OUTPUT_DIR = Path("evaluation")
K = 3  # nombre d'événements fournis au LLM, comme /ask par défaut
JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o-mini")  # LLM qui note les réponses
EMBEDDING_MODEL = "text-embedding-3-small"  # utilisé par answer_relevancy
MAX_CONCURRENCY = 4  # appels simultanés au juge : limite les erreurs 429 de quota

# Seuil minimal attendu pour la moyenne de chaque métrique
THRESHOLDS = {
    "faithfulness": 0.8,
    "answer_relevancy": 0.7,
    "context_precision": 0.7,
    "context_recall": 0.7,
    "factual_correctness": 0.5,
    "retrieval_hit": 0.8,
}


def run_rag(engine: RagEngine, samples: list[dict]) -> list[dict]:
    """Interroge le RAG pour chaque question et garde la réponse et les contextes utilisés."""
    rows = []
    for i, sample in enumerate(samples, 1):
        hits = engine.search(sample["question"], k=K)
        response = engine.generate(sample["question"], hits)
        rows.append({
            "question": sample["question"],
            "reference": sample["reference"],
            "response": response,
            "contexts": [doc.page_content for doc, _ in hits],
            # métrique simple, sans LLM : l'événement attendu a-t-il été retrouvé ?
            "retrieval_hit": float(sample["event_id"] in {doc.metadata["event_id"] for doc, _ in hits}),
        })
        print(f"  [{i}/{len(samples)}] {sample['question']}")
    return rows


async def score_rows(rows: list[dict]) -> None:
    """Calcule les métriques Ragas pour chaque ligne (en parallèle, avec une limite)."""
    client = AsyncOpenAI()  # lit OPENAI_API_KEY dans l'environnement
    llm = llm_factory(JUDGE_MODEL, client=client)
    embeddings = OpenAIEmbeddings(client=client, model=EMBEDDING_MODEL)

    # Chaque métrique : (nom, objet Ragas, fonction qui construit ses arguments)
    metrics = [
        ("faithfulness", Faithfulness(llm=llm),
         lambda r: dict(user_input=r["question"], response=r["response"], retrieved_contexts=r["contexts"])),
        ("answer_relevancy", AnswerRelevancy(llm=llm, embeddings=embeddings),
         lambda r: dict(user_input=r["question"], response=r["response"])),
        ("context_precision", ContextPrecision(llm=llm),
         lambda r: dict(user_input=r["question"], reference=r["reference"], retrieved_contexts=r["contexts"])),
        ("context_recall", ContextRecall(llm=llm),
         lambda r: dict(user_input=r["question"], reference=r["reference"], retrieved_contexts=r["contexts"])),
        ("factual_correctness", FactualCorrectness(llm=llm),
         lambda r: dict(response=r["response"], reference=r["reference"])),
    ]
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def score_one(row, name, metric, make_args):
        async with semaphore:
            try:
                result = await metric.ascore(**make_args(row))
                row[name] = float(result.value)
            except Exception as exc:  # une erreur sur un échantillon ne bloque pas les autres
                row[name] = math.nan
                print(f"  ! {name} a échoué pour « {row['question'][:40]} » : {exc}")

    await asyncio.gather(*(
        score_one(row, name, metric, make_args)
        for row in rows
        for name, metric, make_args in metrics
    ))


def main() -> int:
    # --help affiche la docstring du module sans lancer l'évaluation (payante)
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY absente du .env : impossible d'appeler le LLM.")
        return 1
    samples = json.loads(DATASET.read_text(encoding="utf-8"))

    print(f"1. Génération des réponses du RAG ({len(samples)} questions)")
    rows = run_rag(RagEngine(), samples)

    print(f"2. Notation Ragas (juge : {JUDGE_MODEL})")
    asyncio.run(score_rows(rows))

    # --- Résultats ---
    df = pd.DataFrame(rows)
    metric_names = list(THRESHOLDS)
    df[["question", "response", "contexts", "reference", *metric_names]].to_json(
        OUTPUT_DIR / "ragas_results.json", orient="records", force_ascii=False, indent=2
    )
    df[["question", *metric_names]].to_csv(OUTPUT_DIR / "ragas_results.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nScores par question :")
    print(df[["question", *metric_names]].round(2).to_string(index=False, max_colwidth=45))

    print("\nMoyennes :")
    failed = []
    for name in metric_names:
        mean = df[name].mean()  # les NaN (échecs) sont ignorés
        ok = mean >= THRESHOLDS[name]
        failed += [] if ok else [name]
        print(f"  {'OK   ' if ok else 'ÉCHEC'} {name:<20} {mean:.2f}  (seuil {THRESHOLDS[name]})")

    print(f"\nDétail (réponses, contextes) : {OUTPUT_DIR / 'ragas_results.json'}")
    if failed:
        print(f"Métriques sous le seuil : {', '.join(failed)}")
        return 1
    print("Toutes les métriques atteignent leur seuil.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

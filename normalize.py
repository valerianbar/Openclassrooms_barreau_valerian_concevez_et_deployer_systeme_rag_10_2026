import json
import re
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, field_validator

INPUT = Path("data/events.json")
MAX_DESCRIPTION_CHARS = 1000  # au-delà, la description est découpée en chunks
CHUNK_OVERLAP = 100


class PriceStatus(StrEnum):
    FREE = "free"
    PAID = "paid"
    UNKNOWN = "unknown"  # information absente ou ambiguë : ne jamais lire comme « gratuit »


class Price(BaseModel):
    status: PriceStatus = PriceStatus.UNKNOWN
    details: str | None = None  # texte d'origine nettoyé, conservé pour l'affichage


class Event(BaseModel):
    id: str
    title: str
    description: str | None = None
    start: datetime | None = None
    end: datetime | None = None
    venue: str | None = None
    address: str | None = None
    city: str | None = None
    postal_code: str | None = None
    department: str | None = None
    region: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    category: str | None = None
    keywords: list[str] = []
    price: Price = Price()
    url: str | None = None

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("titre vide")
        return v


# ---------------------------------------------------------------- nettoyage


def clean_html(raw: str | None) -> str | None:
    """Retire le HTML en gardant les sauts de paragraphe/liste, normalise les espaces."""
    if not raw:
        return None
    soup = BeautifulSoup(raw, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for li in soup.find_all("li"):
        li.insert_before("\n- ")
    text = soup.get_text("\n" if soup.find(["p", "div"]) else "")
    text = text.replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return text or None


def clean_text(raw: str | None) -> str | None:
    text = clean_html(raw)
    return re.sub(r"\s+", " ", text) if text else None


_FREE = re.compile(r"^(l')?(entrée\s+)?(libre|gratuite?)(\s+et\s+(libre|gratuite?))?\.?$", re.I)
_PAID = re.compile(r"\d\s*(€|euros?|chf|\$|£)|(€|chf)\s*\d", re.I)


def parse_price(conditions: str | None) -> Price:
    """Gratuit seulement si le texte l'affirme sans ambiguïté ; payant si un montant apparaît.

    « Inscription gratuite mais obligatoire » ou un texte sans montant restent UNKNOWN.
    """
    text = clean_text(conditions)
    if not text:
        return Price()
    if _FREE.match(text.strip()):
        return Price(status=PriceStatus.FREE, details=text)
    if _PAID.search(text):
        return Price(status=PriceStatus.PAID, details=text)
    return Price(details=text)


# ------------------------------------------------------------- normalisation


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def normalize_event(raw: dict) -> Event:
    long_desc = clean_html(raw.get("longdescription_fr"))
    short_desc = clean_html(raw.get("description_fr"))
    description = long_desc or short_desc
    if long_desc and short_desc and short_desc not in long_desc:
        description = f"{short_desc}\n\n{long_desc}"

    coords = raw.get("location_coordinates") or {}
    category = raw.get("category")
    if isinstance(category, list):
        category = ", ".join(map(str, category))

    return Event(
        id=str(raw["uid"]),
        title=clean_text(raw.get("title_fr")) or "",
        description=description,
        start=_parse_dt(raw.get("firstdate_begin")),
        end=_parse_dt(raw.get("lastdate_end")),
        venue=clean_text(raw.get("location_name")),
        address=clean_text(raw.get("location_address")),
        city=clean_text(raw.get("location_city")),
        postal_code=clean_text(raw.get("location_postalcode")),
        department=clean_text(raw.get("location_department")),
        region=clean_text(raw.get("location_region")),
        latitude=coords.get("lat"),
        longitude=coords.get("lon"),
        category=clean_text(category),
        keywords=[k for k in (clean_text(k) for k in raw.get("keywords_fr") or []) if k],
        price=parse_price(raw.get("conditions_fr")),
        url=raw.get("canonicalurl"),
    )


# ----------------------------------------------------------------- Documents


def _fmt_date(dt: datetime | None) -> str | None:
    return dt.strftime("%d/%m/%Y %H:%M") if dt else None


def _header(event: Event) -> str:
    when = _fmt_date(event.start)
    if when and event.end and event.end != event.start:
        when += f" → {_fmt_date(event.end)}"
    place_parts: list[str] = []
    for part in (event.venue, event.address, event.city):
        if part and not any(part in seen for seen in place_parts):
            place_parts.append(part)
    place = ", ".join(place_parts)
    if event.price.status is PriceStatus.FREE:
        price = "Gratuit"
    elif event.price.details:
        price = event.price.details
    else:
        price = "Non précisé"
    lines = [
        f"Titre : {event.title}",
        f"Quand : {when}" if when else None,
        f"Lieu : {place}" if place else None,
        f"Catégorie : {event.category}" if event.category else None,
        f"Mots-clés : {', '.join(event.keywords)}" if event.keywords else None,
        f"Tarif : {price}",
        f"Source : {event.url}" if event.url else None,
    ]
    return "\n".join(line for line in lines if line)


def _metadata(event: Event) -> dict:
    """Métadonnées plates (scalaires uniquement), compatibles avec les filtres des vector stores."""
    meta = {
        "event_id": event.id,
        "title": event.title,
        "start": event.start.isoformat() if event.start else None,
        "end": event.end.isoformat() if event.end else None,
        "start_ts": int(event.start.timestamp()) if event.start else None,
        "end_ts": int(event.end.timestamp()) if event.end else None,
        "venue": event.venue,
        "city": event.city,
        "postal_code": event.postal_code,
        "department": event.department,
        "region": event.region,
        "latitude": event.latitude,
        "longitude": event.longitude,
        "category": event.category,
        "keywords": ", ".join(event.keywords) or None,
        "price_status": event.price.status.value,
        "price_details": event.price.details,
        "source": event.url,
    }
    return {k: v for k, v in meta.items() if v is not None}


_splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_DESCRIPTION_CHARS, chunk_overlap=CHUNK_OVERLAP)


def event_to_documents(event: Event) -> list[Document]:
    header = _header(event)
    description = event.description
    parts = _splitter.split_text(description) if description else []
    if len(parts) <= 1:  # cas courant : un seul document par événement
        text = f"{header}\n\nDescription : {description}" if description else header
        return [Document(page_content=text, metadata=_metadata(event) | {"chunk_index": 0, "chunk_count": 1})]
    return [
        Document(
            page_content=f"{header}\n\nDescription (partie {i + 1}/{len(parts)}) : {part}",
            metadata=_metadata(event) | {"chunk_index": i, "chunk_count": len(parts)},
        )
        for i, part in enumerate(parts)
    ]


def load_documents(path: Path = INPUT) -> tuple[list[Document], list[str]]:
    documents: list[Document] = []
    errors: list[str] = []
    for raw in json.loads(path.read_text(encoding="utf-8")):
        try:
            documents.extend(event_to_documents(normalize_event(raw)))
        except Exception as exc:  # un événement invalide ne doit pas bloquer les autres
            errors.append(f"{raw.get('uid')}: {exc}")
    return documents, errors


def main():
    documents, errors = load_documents()
    events = {d.metadata["event_id"] for d in documents}
    first_chunks = [d for d in documents if d.metadata["chunk_index"] == 0]  # un par événement
    prices = {s: sum(d.metadata["price_status"] == s for d in first_chunks) for s in PriceStatus}
    print(f"{len(events)} événements → {len(documents)} documents, {len(errors)} rejetés")
    print(f"tarifs (par événement) : {prices}")
    for err in errors:
        print("  rejeté :", err)
    print("\n--- exemple ---")
    print(documents[0].page_content)
    print(documents[0].metadata)


if __name__ == "__main__":
    main()

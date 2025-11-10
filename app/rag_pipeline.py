from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, TypedDict

import numpy as np
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

from .config import BASE_DIR, Settings

VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstores"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


class EmbeddingState(TypedDict, total=False):
    file_path: Path
    metadata: Dict[str, Any]
    raw_text: str
    cleaned_text: str
    chunks: List[str]
    summaries: List[str]
    embeddings: List[List[float]]
    output_path: Path


class EmbeddingPipelineError(Exception):
    """Raised when the embedding pipeline fails to process a document."""


@dataclass(frozen=True)
class EmbeddingArtifacts:
    output_path: Path
    chunk_count: int


def _strip_html(text: str) -> str:
    without_scripts = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    without_styles = re.sub(r"(?is)<style.*?>.*?</style>", " ", without_scripts)
    without_tags = re.sub(r"(?s)<[^>]+>", " ", without_styles)
    condensed = re.sub(r"\s+", " ", without_tags)
    return condensed.strip()


def _load_text(state: EmbeddingState) -> EmbeddingState:
    file_path = state["file_path"]
    try:
        raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        raise EmbeddingPipelineError(f"Unable to read {file_path}: {exc}") from exc
    cleaned = _strip_html(raw_text)
    return {"raw_text": raw_text, "cleaned_text": cleaned}


def _chunk_text(state: EmbeddingState) -> EmbeddingState:
    text = state.get("cleaned_text") or state.get("raw_text") or ""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]
    if not chunks:
        raise EmbeddingPipelineError("Document did not produce any chunks.")
    return {"chunks": chunks}


def _summarize_chunks_factory(settings: Settings):
    llm = ChatBedrock(
        model_id=settings.bedrock_model_id,
        region_name=settings.aws_region,
    )

    def _summarize(state: EmbeddingState) -> EmbeddingState:
        chunks = state.get("chunks") or []
        summaries: List[str] = []
        for chunk in chunks:
            prompt = (
                "Summarize the following SEC filing excerpt in 1-2 concise sentences, "
                "highlighting financial or governance insights if present:\n\n"
                f"{chunk}\n"
            )
            response = llm.invoke(prompt)
            summary = getattr(response, "content", str(response)).strip()
            summaries.append(summary or "No summary generated.")
        return {"summaries": summaries}

    return _summarize


def _embed_chunks_factory(settings: Settings):
    embeddings_model = BedrockEmbeddings(
        model_id=settings.bedrock_embedding_model_id,
        region_name=settings.aws_region,
    )

    def _embed(state: EmbeddingState) -> EmbeddingState:
        chunks = state.get("chunks") or []
        embeddings = embeddings_model.embed_documents(chunks)
        return {"embeddings": embeddings}

    return _embed


def _persist_results(state: EmbeddingState) -> EmbeddingState:
    file_path = state["file_path"]
    metadata = state.get("metadata", {})
    chunks = state.get("chunks") or []
    summaries = state.get("summaries") or []
    embeddings = state.get("embeddings") or []

    if not (len(chunks) == len(summaries) == len(embeddings)):
        raise EmbeddingPipelineError("Chunk data, summaries, and embeddings are misaligned.")

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = VECTORSTORE_DIR / f"{file_path.stem}_vectorstore.jsonl"

    records = []
    for index, (chunk, summary, embedding) in enumerate(zip(chunks, summaries, embeddings)):
        record = {
            "chunk_id": f"{index:04d}",
            "text": chunk,
            "summary": summary,
            "embedding": embedding,
            "metadata": {**metadata, "chunk_index": index},
        }
        records.append(record)

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

    return {"output_path": output_path}


def build_embedding_graph(settings: Settings):
    graph = StateGraph(EmbeddingState)
    graph.add_node("load_text", _load_text)
    graph.add_node("chunk_text", _chunk_text)
    graph.add_node("summarize_chunks", _summarize_chunks_factory(settings))
    graph.add_node("embed_chunks", _embed_chunks_factory(settings))
    graph.add_node("persist", _persist_results)

    graph.add_edge(START, "load_text")
    graph.add_edge("load_text", "chunk_text")
    graph.add_edge("chunk_text", "summarize_chunks")
    graph.add_edge("summarize_chunks", "embed_chunks")
    graph.add_edge("embed_chunks", "persist")
    graph.add_edge("persist", END)
    return graph.compile()


def run_embedding_pipeline(
    file_path: Path,
    *,
    metadata: Dict[str, Any] | None,
    settings: Settings,
) -> EmbeddingArtifacts:
    if not file_path.exists():
        raise EmbeddingPipelineError(f"File not found: {file_path}")

    graph = build_embedding_graph(settings)
    initial_state: EmbeddingState = {
        "file_path": file_path,
        "metadata": metadata or {},
    }
    result = graph.invoke(initial_state)
    output_path = result.get("output_path")
    chunks = result.get("chunks") or []
    if not isinstance(output_path, Path):
        raise EmbeddingPipelineError("Pipeline did not produce an output path.")
    return EmbeddingArtifacts(output_path=output_path, chunk_count=len(chunks))


class RagQueryError(Exception):
    """Raised when a stored vector query cannot be completed."""


def _load_vector_records(path: Path) -> List[dict]:
    if not path.exists():
        raise RagQueryError(f"Vector store not found at {path}")
    records: List[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                embedding = record.get("embedding")
                if embedding is None:
                    continue
                record["embedding"] = [float(value) for value in embedding]
                records.append(record)
    except (OSError, json.JSONDecodeError) as exc:
        raise RagQueryError(f"Unable to load vector store {path}: {exc}") from exc
    if not records:
        raise RagQueryError(f"Vector store {path} is empty.")
    return records


def _rank_vector_records(
    query_embedding: Sequence[float],
    records: Sequence[dict],
    *,
    top_k: int,
) -> List[dict]:
    query_vector = np.array(query_embedding, dtype=np.float64)
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return list(records[:top_k])
    scored: List[tuple[float, dict]] = []
    for record in records:
        embedding = np.array(record.get("embedding", []), dtype=np.float64)
        denom = np.linalg.norm(embedding)
        if denom == 0:
            score = 0.0
        else:
            score = float(np.dot(query_vector, embedding) / (query_norm * denom))
        scored.append((score, record))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in scored[:top_k]]


def query_vectorstore(
    vectorstore_path: Path,
    question: str,
    *,
    settings: Settings,
    top_k: int = 4,
) -> str:
    if not question.strip():
        raise RagQueryError("Question must be a non-empty string.")

    embeddings_model = BedrockEmbeddings(
        model_id=settings.bedrock_embedding_model_id,
        region_name=settings.aws_region,
    )
    records = _load_vector_records(vectorstore_path)
    query_embedding = embeddings_model.embed_query(question)
    top_records = _rank_vector_records(query_embedding, records, top_k=top_k)

    context_sections: List[str] = []
    for record in top_records:
        metadata = record.get("metadata", {})
        summary = record.get("summary") or ""
        text = record.get("text") or ""
        chunk_id = metadata.get("chunk_index")
        source_url = metadata.get("source_url") or vectorstore_path.stem
        header = f"Chunk {chunk_id} (source: {source_url})"
        section = f"{header}\nSummary: {summary}\nContent: {text}"
        context_sections.append(section.strip())

    if not context_sections:
        raise RagQueryError("No relevant chunks found in vector store.")

    context = "\n\n---\n\n".join(context_sections)
    llm = ChatBedrock(
        model_id=settings.bedrock_model_id,
        region_name=settings.aws_region,
    )
    prompt = (
        "You are analyzing SEC filing excerpts retrieved via embeddings. "
        "Use only the provided context to answer the question in a concise, well-structured format. "
        "If the context is insufficient, state that explicitly.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))



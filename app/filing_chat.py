from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, List, Sequence

import boto3
import numpy as np
import requests
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .sec import SecFiling, build_sec_request_headers

logger = logging.getLogger(__name__)

FILING_VECTOR_TABLE_NAME = "agentcore_filing_vectors"
FILING_VECTOR_TTL_SECONDS = 3 * 60 * 60  # 3 hours
MAX_CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
TOP_K = 4


@dataclass
class FilingChatSession:
    session_id: str
    cik: str
    filing: SecFiling
    chunk_count: int


class FilingChatError(Exception):
    """Raised when a filing cannot be prepared for chat."""


def prepare_filing_chat(filing: SecFiling, *, cik: str, settings: Settings) -> FilingChatSession:
    document_text = _download_filing_document(filing.url, timeout=settings.http_timeout)
    if not document_text:
        raise FilingChatError("Filing document is empty.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(document_text)
    if not chunks:
        raise FilingChatError("Unable to split filing into chunks.")

    embeddings_model = BedrockEmbeddings(
        model_id=settings.bedrock_embedding_model_id,
        region_name=settings.aws_region,
    )
    chunk_embeddings = embeddings_model.embed_documents(chunks)

    session_id = uuid.uuid4().hex
    _store_chunks(session_id, chunks, chunk_embeddings, cik, filing, settings)

    return FilingChatSession(
        session_id=session_id,
        cik=cik,
        filing=filing,
        chunk_count=len(chunks),
    )


def answer_filing_question(
    session: FilingChatSession,
    question: str,
    *,
    settings: Settings,
) -> str:
    if not question.strip():
        return "Please provide a question to discuss the filing."

    embeddings_model = BedrockEmbeddings(
        model_id=settings.bedrock_embedding_model_id,
        region_name=settings.aws_region,
    )
    query_embedding = embeddings_model.embed_query(question)

    stored_chunks = _load_chunks(session.session_id, settings)
    if not stored_chunks:
        return (
            "The filing context is no longer available. "
            "Please restart the filing chat workflow."
        )

    ranked_chunks = _rank_chunks(query_embedding, stored_chunks)
    context_snippets = [
        chunk["text"]
        for chunk in ranked_chunks[:TOP_K]
    ]
    context = "\n\n---\n\n".join(context_snippets)

    if not context:
        return "I could not locate relevant context within the filing for that question."

    llm = ChatBedrock(
        model_id=settings.bedrock_model_id,
        region_name=settings.aws_region,
    )
    system_prompt = (
        f"You are assisting with SEC filing chat sessions. "
        f"Focus on the provided filing context only. "
        f"The filing is a {session.filing.form} dated {session.filing.date} "
        f"for CIK {session.cik}. "
        "If you are unsure, say so explicitly."
    )
    user_prompt = (
        "Use the filing context to answer the user's question.\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    return getattr(response, "content", str(response))


def _download_filing_document(url: str, *, timeout: float) -> str:
    try:
        response = requests.get(url, headers=build_sec_request_headers(), timeout=timeout)
    except requests.RequestException as exc:
        raise FilingChatError(f"Unable to download filing document: {exc}") from exc

    if response.status_code != 200:
        if response.status_code == 403:
            raise FilingChatError(
                "SEC filing download failed with status 403. Set SEC_USER_AGENT with valid contact details per "
                f"SEC guidelines for {url}."
            )
        raise FilingChatError(
            f"SEC filing download failed with status {response.status_code} for {url}."
        )

    response.encoding = response.encoding or "utf-8"
    return response.text


def _store_chunks(
    session_id: str,
    chunks: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    cik: str,
    filing: SecFiling,
    settings: Settings,
) -> None:
    if len(chunks) != len(embeddings):
        raise FilingChatError("Chunk embeddings are misaligned.")

    table = _ensure_vector_table(settings)
    expires_at = int(time.time()) + FILING_VECTOR_TTL_SECONDS

    with table.batch_writer() as batch:
        for index, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            vector = [_to_decimal(value) for value in embedding]
            item = {
                "session_id": session_id,
                "chunk_id": f"{index:04d}",
                "text": chunk_text,
                "embedding": vector,
                "cik": cik,
                "form": filing.form,
                "filing_date": filing.date,
                "source_url": filing.url,
                "expires_at": expires_at,
            }
            batch.put_item(Item=item)


def _load_chunks(session_id: str, settings: Settings) -> List[dict]:
    table = _ensure_vector_table(settings)
    try:
        response = table.query(
            KeyConditionExpression=Key("session_id").eq(session_id),
        )
    except ClientError as exc:
        logger.exception("Failed querying filing chunks", exc_info=exc)
        return []
    return response.get("Items", [])


def _rank_chunks(
    query_embedding: Sequence[float],
    items: Iterable[dict],
) -> List[dict]:
    query_vector = np.array(query_embedding, dtype=np.float64)
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return list(items)

    scored: List[tuple[float, dict]] = []
    for item in items:
        embedding = np.array([float(value) for value in item.get("embedding", [])], dtype=np.float64)
        denom = np.linalg.norm(embedding)
        if denom == 0:
            score = 0.0
        else:
            score = float(np.dot(query_vector, embedding) / (query_norm * denom))
        scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored]


def _ensure_vector_table(settings: Settings):
    dynamodb = boto3.resource("dynamodb", region_name=settings.aws_region)
    table = dynamodb.Table(FILING_VECTOR_TABLE_NAME)
    try:
        table.load()
        return table
    except dynamodb.meta.client.exceptions.ResourceNotFoundException:  # type: ignore[attr-defined]
        pass

    logger.info("Creating DynamoDB table %s for filing embeddings", FILING_VECTOR_TABLE_NAME)
    table = dynamodb.create_table(
        TableName=FILING_VECTOR_TABLE_NAME,
        KeySchema=[
            {"AttributeName": "session_id", "KeyType": "HASH"},
            {"AttributeName": "chunk_id", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "session_id", "AttributeType": "S"},
            {"AttributeName": "chunk_id", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    table.wait_until_exists()
    _ensure_ttl(FILING_VECTOR_TABLE_NAME, settings)
    return table


def _ensure_ttl(table_name: str, settings: Settings) -> None:
    client = boto3.client("dynamodb", region_name=settings.aws_region)
    try:
        description = client.describe_time_to_live(TableName=table_name)
        ttl_status = description.get("TimeToLiveDescription", {}).get("TimeToLiveStatus")
        if ttl_status in {"ENABLED", "ENABLING"}:
            return
    except ClientError:
        pass

    client.update_time_to_live(
        TableName=table_name,
        TimeToLiveSpecification={
            "Enabled": True,
            "AttributeName": "expires_at",
        },
    )


def _to_decimal(value: float) -> Decimal:
    return Decimal(str(value))



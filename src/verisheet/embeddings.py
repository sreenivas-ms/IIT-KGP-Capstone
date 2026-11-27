"""Embedding providers for retrieval: the shared capstone embedding API or a local model.

The course provides an OpenAI-compatible ``POST /v1/embeddings`` service serving
``bge-small-en-v1.5`` (384-dim). ``EmbeddingAPIEncoder`` calls it; a local
sentence-transformers model of the same weights is kept as an offline fallback.

Both expose ``encode(texts, **kwargs) -> np.ndarray`` (one row per text), which is the only
interface ``DefinitionRetriever`` relies on.

Configuration (environment / ``.env``, see ``verisheet.config.Settings``):
    EMBEDDING_PROVIDER   auto | api | local   (auto = api when base URL and key are set)
    EMBEDDING_API_BASE   base URL from the coordinator, ending in /v1
    EMBEDDING_API_KEY    shared embedding key from the coordinator (never logged)
    EMBEDDING_MODEL      bge-small-en-v1.5
    EMBEDDING_CA_CERT    path to vm02-ca.crt (private CA); relative paths resolve from the
                         project root, default location config/vm02-ca.crt
"""

from __future__ import annotations

import logging
import math
import ssl
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

MAX_TEXTS_PER_REQUEST = 16  # service limit
MAX_INPUT_CHARS = 1800  # ~450 tokens; service limit is 512 model tokens per text
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
MAX_BACKOFF_SECONDS = 30.0


class EmbeddingServiceError(RuntimeError):
    """The embedding service is unreachable, misconfigured or returned an invalid response.

    Messages never include the API key or request headers.
    """


class EmbeddingAPIEncoder:
    """Client for the shared OpenAI-compatible embeddings endpoint.

    Args:
        base_url: API base ending in ``/v1``.
        api_key: bearer key (kept private; never logged or put in error messages).
        model: served model id, e.g. ``bge-small-en-v1.5``.
        ca_cert: path to the CA certificate that signed the server certificate; ``None`` uses
            system trust. TLS verification is never disabled.
        timeout: per-request timeout in seconds.
        max_attempts: attempts per batch for capacity/transient errors (exponential backoff).
        transport/sleep: injection points for tests.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "bge-small-en-v1.5",
        ca_cert: str | Path | None = None,
        timeout: float = 30.0,
        max_attempts: int = 4,
        transport: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        import httpx  # noqa: PLC0415 - optional at import time of the package

        if not base_url:
            raise EmbeddingServiceError("EMBEDDING_API_BASE is not configured.")
        if not api_key:
            raise EmbeddingServiceError("EMBEDDING_API_KEY is not configured.")
        self.model = model
        self.max_attempts = max(1, max_attempts)
        self._sleep = sleep
        self._url = base_url.rstrip("/") + "/embeddings"
        verify: Any = True
        if ca_cert:
            cert_path = _resolve_path(ca_cert)
            if not cert_path.is_file():
                raise EmbeddingServiceError(
                    f"CA certificate not found at '{cert_path}'. Set EMBEDDING_CA_CERT to the "
                    "path of vm02-ca.crt."
                )
            verify = ssl.create_default_context(cafile=str(cert_path))
        self._client = httpx.Client(
            timeout=timeout,
            verify=verify,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        self._httpx = httpx

    def encode(self, texts: list[str], **_: Any) -> np.ndarray:
        """Embed ``texts`` in batches of 16; returns a ``(len(texts), dim)`` float32 array.

        Extra keyword arguments (``normalize_embeddings`` etc.) are accepted for interface
        compatibility with sentence-transformers; normalisation is done by the caller.
        """
        texts = [str(t)[:MAX_INPUT_CHARS] for t in texts]
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        rows: list[list[float]] = []
        for start in range(0, len(texts), MAX_TEXTS_PER_REQUEST):
            rows.extend(self._embed_batch(texts[start:start + MAX_TEXTS_PER_REQUEST]))
        vectors = np.asarray(rows, dtype=np.float32)
        if vectors.ndim != 2 or len({len(r) for r in rows}) != 1:
            raise EmbeddingServiceError("Embedding service returned vectors of mixed dimension.")
        return vectors

    def close(self) -> None:
        self._client.close()

    # -- internals ----------------------------------------------------------------------------

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": batch}
        last_problem = "no attempt made"
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._client.post(self._url, json=payload)
            except self._httpx.TransportError as exc:  # connect/read timeout, TLS, DNS …
                last_problem = f"{type(exc).__name__}: {_short(str(exc))}"
                if isinstance(exc, self._httpx.ConnectError) and "CERTIFICATE" in str(exc).upper():
                    raise EmbeddingServiceError(
                        f"TLS verification failed ({last_problem}); check EMBEDDING_CA_CERT."
                    ) from exc
            else:
                if response.status_code == 200:
                    return self._parse(response, len(batch))
                if response.status_code in (401, 403):
                    raise EmbeddingServiceError(
                        f"Embedding API rejected the credentials (HTTP {response.status_code}); "
                        "check EMBEDDING_API_KEY."
                    )
                last_problem = f"HTTP {response.status_code}: {_short(response.text)}"
                if response.status_code not in RETRYABLE_STATUS:
                    raise EmbeddingServiceError(f"Embedding API error ({last_problem}).")
                if attempt < self.max_attempts:
                    delay = _retry_after(response.headers.get("Retry-After"), attempt)
                    logger.warning("embedding API busy (%s); retry %d in %.1fs",
                                   response.status_code, attempt, delay)
                    self._sleep(delay)
                    continue
                break
            if attempt < self.max_attempts:
                self._sleep(_retry_after(None, attempt))
        raise EmbeddingServiceError(
            f"Embedding API unavailable after {self.max_attempts} attempts ({last_problem})."
        )

    def _parse(self, response: Any, expected: int) -> list[list[float]]:
        try:
            data = response.json()["data"]
            ordered = sorted(data, key=lambda item: int(item["index"]))
            rows = [[float(x) for x in item["embedding"]] for item in ordered]
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingServiceError("Embedding API returned an unexpected response shape.") from exc
        if len(rows) != expected or [int(i["index"]) for i in ordered] != list(range(expected)):
            raise EmbeddingServiceError(
                f"Embedding API returned {len(rows)} vectors for {expected} inputs."
            )
        if not all(math.isfinite(x) for row in rows for x in row):
            raise EmbeddingServiceError("Embedding API returned non-finite values.")
        return rows


def embedding_provider(settings: Any = None) -> str:
    """Resolve the configured provider: ``"api"`` or ``"local"``."""
    settings = settings or _settings()
    choice = (settings.embedding_provider or "auto").strip().lower()
    if choice in ("api", "local"):
        return choice
    if choice != "auto":
        raise ValueError("EMBEDDING_PROVIDER must be auto, api or local")
    has_key = bool(settings.embedding_api_key.get_secret_value())
    return "api" if settings.embedding_api_base and has_key else "local"


def create_encoder(model_name: str, settings: Any = None) -> Any:
    """Build the encoder for ``model_name`` using the configured provider.

    The API model must be the same model as ``model_name`` (indexing and search must share one
    embedding space); a mismatch raises ``ValueError``.
    """
    settings = settings or _settings()
    if embedding_provider(settings) == "api":
        if _model_key(settings.embedding_model) != _model_key(model_name):
            raise ValueError(
                f"EMBEDDING_MODEL '{settings.embedding_model}' differs from retriever model "
                f"'{model_name}'; indexing and search must use the same model."
            )
        logger.info("embeddings via API model=%s", settings.embedding_model)
        return EmbeddingAPIEncoder(
            base_url=settings.embedding_api_base,
            api_key=settings.embedding_api_key.get_secret_value(),
            model=settings.embedding_model,
            ca_cert=settings.embedding_ca_cert or None,
            timeout=settings.embedding_timeout_seconds,
        )
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415 - heavy, optional
    except ImportError as exc:
        raise ImportError("Local embeddings need `pip install -r requirements-local-embeddings.txt`, "
                          "or configure EMBEDDING_API_BASE/EMBEDDING_API_KEY to use the embedding API.") from exc

    logger.info("embeddings via local sentence-transformers model=%s", model_name)
    return SentenceTransformer(model_name)


def _resolve_path(value: str | Path) -> Path:
    """Expand ``~``; resolve a relative path from the CWD, else from the project root."""
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path
    project_root = Path(__file__).resolve().parents[2]
    return project_root / path


def _model_key(name: str) -> str:
    return name.strip().split("/")[-1].lower()


def _retry_after(header: str | None, attempt: int) -> float:
    try:
        if header is not None:
            return min(MAX_BACKOFF_SECONDS, max(0.0, float(header)))
    except ValueError:
        pass
    return min(MAX_BACKOFF_SECONDS, 1.0 * 2 ** (attempt - 1))


def _short(text: str, limit: int = 200) -> str:
    return " ".join(text.split())[:limit]


def _settings() -> Any:
    from verisheet.config import get_settings  # noqa: PLC0415

    return get_settings()

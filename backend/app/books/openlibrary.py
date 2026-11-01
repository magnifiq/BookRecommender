import asyncio
import json
import re
import time
from typing import Any, Literal

import httpx

from app.books.models import BookRecord


class BookProviderUnavailable(RuntimeError):
    """A sanitized error indicating that the metadata provider could not answer."""


class BookProviderNotFound(LookupError):
    """The provider has no record matching an exact identifier."""


class OpenLibraryClient:
    BASE_URL = "https://openlibrary.org"
    SEARCH_FIELDS = (
        "key,title,author_name,first_publish_year,subject,isbn,cover_i,"
        "edition_key,first_sentence,description,number_of_pages_median"
    )

    def __init__(
        self,
        http_client: Any | None = None,
        timeout_seconds: float = 5.0,
        cache_ttl_seconds: float = 60.0,
        min_request_interval_seconds: float = 1.0,
        user_agent: str = "BookRecommender/0.1",
    ):
        self.http_client = http_client
        self.timeout_seconds = timeout_seconds
        self.cache_ttl_seconds = cache_ttl_seconds
        self.min_request_interval_seconds = min_request_interval_seconds
        self.user_agent = user_agent
        self._cache: dict[str, tuple[float, dict]] = {}
        self._request_lock = asyncio.Lock()
        self._last_request_at = 0.0

    async def search(
        self,
        query: str,
        search_by: Literal["title", "author", "subject", "any"] = "any",
        limit: int = 5,
    ) -> list[BookRecord]:
        params: dict[str, Any] = {"limit": limit, "fields": self.SEARCH_FIELDS}
        if search_by == "any":
            params["q"] = query
        else:
            params[search_by] = query
        payload = await self._get_json("/search.json", params)
        docs = payload.get("docs", [])
        if not isinstance(docs, list):
            raise BookProviderUnavailable("Open Library returned an invalid search response")
        return [book for item in docs if isinstance(item, dict)
                if (book := self._normalize_search_record(item)) is not None]

    async def get_details(self, identifier: str) -> BookRecord | None:
        value = identifier.strip()
        if re.fullmatch(r"OL\d+W", value, re.IGNORECASE):
            payload = await self._get_json(f"/works/{value.upper()}.json")
            work = self._normalize_work(payload, value.upper())
            return await self._resolve_author_names(work, payload.get("authors", []))
        if re.fullmatch(r"OL\d+M", value, re.IGNORECASE):
            payload = await self._get_json(f"/books/{value.upper()}.json")
            edition = self._normalize_edition(payload, value.upper())
            work_ids = payload.get("works") or []
            if edition and work_ids and isinstance(work_ids[0], dict):
                key = work_ids[0].get("key", "")
                work_id = key.rsplit("/", 1)[-1]
                try:
                    work_payload = await self._get_json(f"/works/{work_id}.json")
                    work = self._normalize_work(work_payload, work_id)
                    work = await self._resolve_author_names(work, work_payload.get("authors", []))
                    return self._merge_work_and_edition(work, edition) if work else edition
                except BookProviderUnavailable:
                    pass
            return await self._resolve_author_names(edition, payload.get("authors", []))
        if re.fullmatch(r"[0-9Xx-]{10,17}", value):
            normalized_isbn = value.replace("-", "")
            payload = await self._get_json(f"/isbn/{normalized_isbn}.json")
            edition_id = self._path_id(payload.get("key"))
            edition = self._normalize_edition(payload, edition_id)
            work_ids = payload.get("works") or []
            if edition and work_ids and isinstance(work_ids[0], dict):
                work_id = self._path_id(work_ids[0].get("key"))
                try:
                    work_payload = await self._get_json(f"/works/{work_id}.json")
                    work = self._normalize_work(work_payload, work_id)
                    work = await self._resolve_author_names(work, work_payload.get("authors", []))
                    return self._merge_work_and_edition(work, edition) if work else edition
                except BookProviderUnavailable:
                    pass
            return await self._resolve_author_names(edition, payload.get("authors", []))

        matches = await self.search(value, search_by="title", limit=1)
        if not matches:
            return None
        if matches[0].work_id:
            try:
                work_payload = await self._get_json(f"/works/{matches[0].work_id}.json")
                work = self._normalize_work(
                    work_payload, matches[0].work_id, fallback=matches[0]
                )
                return work or matches[0]
            except BookProviderUnavailable:
                return matches[0]
        return matches[0]

    async def _resolve_author_names(self, book: BookRecord | None, authors: Any) -> BookRecord | None:
        if book is None or book.authors or not isinstance(authors, list):
            return book
        author_keys = []
        for item in authors[:3]:
            if not isinstance(item, dict):
                continue
            author = item.get("author") if isinstance(item.get("author"), dict) else item
            key = author.get("key") if isinstance(author, dict) else None
            author_id = self._path_id(key)
            if author_id:
                author_keys.append(author_id)
        if not author_keys:
            return book
        payloads = await asyncio.gather(
            *(self._get_json(f"/authors/{author_id}.json") for author_id in author_keys),
            return_exceptions=True,
        )
        names = [payload.get("name") for payload in payloads
                 if isinstance(payload, dict) and isinstance(payload.get("name"), str)]
        return book.model_copy(update={"authors": self._strings(names)}) if names else book

    async def _get_json(self, path: str, params: dict | None = None) -> dict:
        cache_key = path + "?" + json.dumps(params or {}, sort_keys=True)
        now = time.monotonic()
        cached = self._cache.get(cache_key)
        if cached and now - cached[0] < self.cache_ttl_seconds:
            return cached[1]

        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        try:
            async with self._request_lock:
                cached = self._cache.get(cache_key)
                if cached and time.monotonic() - cached[0] < self.cache_ttl_seconds:
                    return cached[1]
                delay = self.min_request_interval_seconds - (
                    time.monotonic() - self._last_request_at
                )
                if delay > 0:
                    await asyncio.sleep(delay)
                self._last_request_at = time.monotonic()
                if self.http_client is not None:
                    response = await self.http_client.get(
                        self.BASE_URL + path,
                        params=params,
                        headers=headers,
                        timeout=self.timeout_seconds,
                    )
                else:
                    async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                        response = await client.get(
                            self.BASE_URL + path, params=params, headers=headers
                        )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise BookProviderUnavailable("Open Library returned an invalid response")
        except httpx.TimeoutException as exc:
            raise BookProviderUnavailable("Open Library request timed out") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise BookProviderNotFound("No matching book was found") from exc
            raise BookProviderUnavailable(
                f"Open Library returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise BookProviderUnavailable("Open Library is unavailable") from exc
        except (ValueError, TypeError) as exc:
            raise BookProviderUnavailable("Open Library returned invalid JSON") from exc

        if self.cache_ttl_seconds > 0:
            if len(self._cache) >= 256:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = (time.monotonic(), payload)
        return payload

    def _normalize_search_record(self, item: dict) -> BookRecord | None:
        title = item.get("title")
        if not isinstance(title, str) or not title.strip():
            return None
        work_id = self._path_id(item.get("key"))
        cover_id = self._as_int(item.get("cover_i"))
        first_sentence = self._description(item.get("first_sentence"))
        description = self._description(item.get("description")) or first_sentence
        edition_ids = item.get("edition_key") or []
        return BookRecord(
            work_id=work_id,
            edition_id=edition_ids[0] if edition_ids else None,
            title=title.strip(),
            authors=self._strings(item.get("author_name")),
            description=self._limit_text(description),
            publication_year=self._as_int(item.get("first_publish_year")),
            subjects=self._strings(item.get("subject"), limit=12),
            isbns=self._strings(item.get("isbn"), limit=5),
            cover_id=cover_id,
            cover_url=self._cover_url(cover_id),
            page_count=self._as_int(item.get("number_of_pages_median")),
            provider_url=f"{self.BASE_URL}{item['key']}" if item.get("key") else None,
        )

    def _normalize_work(self, item: dict, work_id: str,
                        fallback: BookRecord | None = None) -> BookRecord | None:
        title = item.get("title") or (fallback.title if fallback else None)
        if not isinstance(title, str) or not title.strip():
            return None
        cover_id = self._first_int(item.get("covers"))
        authors = self._author_names(item.get("authors")) or (fallback.authors if fallback else [])
        return BookRecord(
            work_id=work_id,
            edition_id=fallback.edition_id if fallback else None,
            title=title.strip(),
            authors=authors,
            description=self._limit_text(self._description(item.get("description"))
                                         or (fallback.description if fallback else None)),
            publication_year=self._year(item.get("first_publish_date"))
                              or (fallback.publication_year if fallback else None),
            subjects=self._strings(item.get("subjects"), limit=12)
                     or (fallback.subjects if fallback else []),
            isbns=fallback.isbns if fallback else [],
            cover_id=cover_id or (fallback.cover_id if fallback else None),
            cover_url=self._cover_url(cover_id or (fallback.cover_id if fallback else None)),
            page_count=fallback.page_count if fallback else None,
            provider_url=f"{self.BASE_URL}/works/{work_id}",
        )

    def _normalize_edition(self, item: dict, edition_id: str | None) -> BookRecord | None:
        title = item.get("title")
        if not isinstance(title, str) or not title.strip():
            return None
        cover_id = self._first_int(item.get("covers"))
        work_id = None
        works = item.get("works") or []
        if works and isinstance(works[0], dict):
            work_id = self._path_id(works[0].get("key"))
        isbns = self._strings(item.get("isbn_13")) + self._strings(item.get("isbn_10"))
        return BookRecord(
            work_id=work_id,
            edition_id=edition_id,
            title=title.strip(),
            authors=self._author_names(item.get("authors")),
            description=self._limit_text(self._description(item.get("description"))),
            publication_year=self._year(item.get("publish_date")),
            subjects=self._strings(item.get("subjects"), limit=12),
            isbns=isbns[:5],
            cover_id=cover_id,
            cover_url=self._cover_url(cover_id),
            page_count=self._as_int(item.get("number_of_pages")),
            provider_url=f"{self.BASE_URL}/books/{edition_id}" if edition_id else None,
        )

    @staticmethod
    def _merge_work_and_edition(work: BookRecord, edition: BookRecord) -> BookRecord:
        return work.model_copy(update={
            "edition_id": edition.edition_id,
            "authors": work.authors or edition.authors,
            "description": work.description or edition.description,
            "publication_year": edition.publication_year or work.publication_year,
            "subjects": work.subjects or edition.subjects,
            "isbns": edition.isbns,
            "cover_id": edition.cover_id or work.cover_id,
            "cover_url": edition.cover_url or work.cover_url,
            "page_count": edition.page_count,
        })

    @staticmethod
    def _path_id(path: Any) -> str | None:
        if not isinstance(path, str):
            return None
        return path.rstrip("/").rsplit("/", 1)[-1]

    @staticmethod
    def _description(value: Any) -> str | None:
        if isinstance(value, dict):
            value = value.get("value")
        if isinstance(value, list):
            value = " ".join(str(part) for part in value if part)
        return value.strip() if isinstance(value, str) and value.strip() else None

    @staticmethod
    def _strings(values: Any, limit: int = 10) -> list[str]:
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            return []
        return [value.strip() for value in values
                if isinstance(value, str) and value.strip()][:limit]

    @classmethod
    def _author_names(cls, values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        names = []
        for item in values:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                name = item.get("name")
                author = item.get("author")
                if isinstance(author, dict):
                    name = name or author.get("name")
                if isinstance(name, str):
                    names.append(name)
        return cls._strings(names, limit=10)

    @staticmethod
    def _as_int(value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _first_int(values: Any) -> int | None:
        return OpenLibraryClient._as_int(values[0]) if isinstance(values, list) and values else None

    @staticmethod
    def _year(value: Any) -> int | None:
        if not isinstance(value, str):
            return None
        match = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", value)
        return int(match.group(1)) if match else None

    @staticmethod
    def _limit_text(value: str | None, max_length: int = 1200) -> str | None:
        if not value:
            return None
        return value[:max_length].rstrip()

    @staticmethod
    def _cover_url(cover_id: int | None) -> str | None:
        return f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg?default=false" if cover_id else None

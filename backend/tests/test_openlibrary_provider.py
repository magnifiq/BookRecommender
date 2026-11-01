import httpx
import pytest

from app.books.openlibrary import (
    BookProviderNotFound,
    BookProviderUnavailable,
    OpenLibraryClient,
)


@pytest.mark.asyncio
async def test_search_uses_requested_field_and_normalizes_provider_data():
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"docs": [{
            "key": "/works/OL1W",
            "title": "Dune",
            "author_name": ["Frank Herbert"],
            "first_publish_year": 1965,
            "subject": ["Science fiction", "Politics"],
            "isbn": ["9780441172719"],
            "cover_i": 123,
            "edition_key": ["OL2M"],
            "first_sentence": ["A beginning sentence."],
            "number_of_pages_median": 412,
        }]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenLibraryClient(client, min_request_interval_seconds=0)
        books = await provider.search("Frank Herbert", search_by="author", limit=3)
    assert requests[0].url.params["author"] == "Frank Herbert"
    assert requests[0].headers["user-agent"].startswith("BookRecommender")
    assert len(books) == 1
    assert books[0].title == "Dune"
    assert books[0].authors == ["Frank Herbert"]
    assert books[0].publication_year == 1965
    assert books[0].cover_url == "https://covers.openlibrary.org/b/id/123-M.jpg?default=false"
    assert books[0].description == "A beginning sentence."


@pytest.mark.asyncio
async def test_successful_search_is_cached():
    count = 0
    def handler(_request):
        nonlocal count
        count += 1
        return httpx.Response(200, json={"docs": []})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenLibraryClient(client, min_request_interval_seconds=0)
        assert await provider.search("no results") == []
        assert await provider.search("no results") == []
    assert count == 1


@pytest.mark.asyncio
async def test_details_merge_work_description_and_edition_fields():
    paths = []
    def handler(request):
        paths.append(request.url.path)
        if request.url.path == "/books/OL2M.json":
            return httpx.Response(200, json={
                "key": "/books/OL2M", "title": "Dune", "authors": [{"name": "Frank Herbert"}],
                "publish_date": "1965", "isbn_13": ["9780441172719"], "covers": [234],
                "number_of_pages": 412, "works": [{"key": "/works/OL1W"}],
            })
        return httpx.Response(200, json={
            "title": "Dune", "description": {"value": "A political science fiction novel."},
            "subjects": ["Science fiction"], "first_publish_date": "1965",
        })
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenLibraryClient(client, min_request_interval_seconds=0)
        book = await provider.get_details("OL2M")
    assert paths == ["/books/OL2M.json", "/works/OL1W.json"]
    assert book.title == "Dune"
    assert book.description == "A political science fiction novel."
    assert book.authors == ["Frank Herbert"]
    assert book.isbns == ["9780441172719"]
    assert book.page_count == 412


@pytest.mark.asyncio
async def test_work_author_references_are_resolved_when_available():
    def handler(request):
        if request.url.path == "/works/OL1W.json":
            return httpx.Response(200, json={
                "title": "Dune", "authors": [{"author": {"key": "/authors/OL9A"}}],
            })
        if request.url.path == "/authors/OL9A.json":
            return httpx.Response(200, json={"name": "Frank Herbert"})
        return httpx.Response(404, request=request)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenLibraryClient(client, min_request_interval_seconds=0)
        book = await provider.get_details("OL1W")
    assert book.authors == ["Frank Herbert"]


@pytest.mark.asyncio
async def test_missing_book_and_provider_http_error_are_distinguished():
    def missing(request):
        return httpx.Response(404, request=request)
    async with httpx.AsyncClient(transport=httpx.MockTransport(missing)) as client:
        provider = OpenLibraryClient(client, min_request_interval_seconds=0)
        with pytest.raises(BookProviderNotFound):
            await provider.get_details("OL123W")

    def unavailable(request):
        return httpx.Response(503, request=request)
    async with httpx.AsyncClient(transport=httpx.MockTransport(unavailable)) as client:
        provider = OpenLibraryClient(client, min_request_interval_seconds=0)
        with pytest.raises(BookProviderUnavailable, match="HTTP 503"):
            await provider.search("books")


@pytest.mark.asyncio
async def test_provider_timeout_is_sanitized():
    def timeout(request):
        raise httpx.ReadTimeout("private request details", request=request)
    async with httpx.AsyncClient(transport=httpx.MockTransport(timeout)) as client:
        provider = OpenLibraryClient(client, min_request_interval_seconds=0)
        with pytest.raises(BookProviderUnavailable, match="timed out") as error:
            await provider.search("books")
    assert "private request details" not in str(error.value)

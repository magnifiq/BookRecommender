import pytest

from app.books.models import BookRecord
from app.books.openlibrary import BookProviderUnavailable
from app.tools.book_details import BookDetailsInput, build_book_details_tool
from app.tools.book_search import BookSearchInput, build_search_books_tool
from app.tools.generate_story import GenerateStoryInput, build_generate_story_tool
from app.tools.recommend_books import RecommendBooksInput, build_recommend_books_tool
from app.tools.summarize_book import SummarizeBookInput, build_summarize_book_tool
from app.tools.base import PermissionPolicy


class FakeProvider:
    def __init__(self, books=None, error=False):
        self.books = books or []
        self.error = error
        self.search_calls = []

    async def search(self, query, search_by="any", limit=5):
        self.search_calls.append((query, search_by, limit))
        if self.error:
            raise BookProviderUnavailable("secret transport error")
        return self.books[:limit]

    async def get_details(self, identifier):
        if self.error:
            raise BookProviderUnavailable("secret transport error")
        return next((book for book in self.books
                     if identifier in {book.title, book.work_id, book.edition_id, *book.isbns}), None)


@pytest.fixture
def books():
    return [
        BookRecord(work_id="OL1W", edition_id="OL2M", title="Dune",
                   authors=["Frank Herbert"], description=(
                       "Paul travels to Arrakis. The desert world holds political danger. "
                       "A new alliance changes his future."),
                   publication_year=1965, subjects=["Science fiction", "Politics"],
                   isbns=["9780441172719"], page_count=412),
        BookRecord(work_id="OL3W", title="The Martian", authors=["Andy Weir"],
                   description="An astronaut survives alone on Mars.",
                   subjects=["Science fiction", "Survival"], page_count=369),
    ]


@pytest.mark.asyncio
async def test_search_tool_returns_normalized_records_and_handles_unavailable(books):
    tool = build_search_books_tool(FakeProvider(books))
    output = await tool.handler(BookSearchInput(query="science fiction", search_by="subject", limit=2))
    assert output["status"] == "ok"
    assert output["books"][0]["title"] == "Dune"
    assert "docs" not in output
    assert tool.permission_policy == PermissionPolicy.AUTO

    unavailable = await build_search_books_tool(FakeProvider(error=True)).handler(
        BookSearchInput(query="Dune")
    )
    assert unavailable["status"] == "unavailable"
    assert "secret" not in str(unavailable)
    assert unavailable["books"] == []


@pytest.mark.asyncio
async def test_details_tool_returns_missing_fields_and_not_found(books):
    tool = build_book_details_tool(FakeProvider(books))
    result = await tool.handler(BookDetailsInput(identifier="OL1W"))
    assert result["status"] == "ok"
    assert result["book"]["title"] == "Dune"
    assert result["book"]["cover_id"] is None

    missing = await tool.handler(BookDetailsInput(identifier="unknown title"))
    assert missing["status"] == "not_found"
    assert missing["book"] is None


@pytest.mark.asyncio
async def test_recommendations_rank_and_exclude_read_books(books):
    provider = FakeProvider(books)
    tool = build_recommend_books_tool(provider)
    result = await tool.handler(RecommendBooksInput(
        genres=["Science fiction"], topics=["Mars"], favorite_authors=["Andy Weir"],
        already_read_or_disliked=["Dune"], limit=5,
    ))
    assert result["status"] == "ok"
    assert [item["book"]["title"] for item in result["recommendations"]] == ["The Martian"]
    assert any("preferred author" in reason for reason in result["recommendations"][0]["reasons"])


@pytest.mark.asyncio
async def test_recommendations_handle_insufficient_preferences_and_no_matches(books):
    provider = FakeProvider(books)
    tool = build_recommend_books_tool(provider)
    insufficient = await tool.handler(RecommendBooksInput())
    assert insufficient["status"] == "insufficient_preferences"
    assert provider.search_calls == []
    no_match = await tool.handler(RecommendBooksInput(topics=["underwater robots"]))
    assert no_match["status"] == "no_matches"


@pytest.mark.asyncio
async def test_summary_marks_source_and_never_claims_comprehensive(books):
    tool = build_summarize_book_tool(FakeProvider(books))
    metadata = await tool.handler(SummarizeBookInput(identifier="Dune", length="short"))
    assert metadata["status"] == "ok"
    assert metadata["basis"] == "metadata_description"
    assert metadata["comprehensive"] is False
    assert "catalog description" in metadata["note"]
    assert len(metadata["summary"].split()) <= 36

    supplied = await tool.handler(SummarizeBookInput(
        supplied_text="First supplied sentence. Second supplied sentence. Third supplied sentence.",
        length="short",
    ))
    assert supplied["basis"] == "user_supplied_text"
    assert "user-supplied text" in supplied["note"]


@pytest.mark.asyncio
async def test_summary_does_not_invent_when_description_is_missing():
    tool = build_summarize_book_tool(FakeProvider([BookRecord(title="Metadata only")]))
    result = await tool.handler(SummarizeBookInput(identifier="Metadata only"))
    assert result["status"] == "no_description"
    assert result["summary"] is None
    assert result["comprehensive"] is False


@pytest.mark.asyncio
async def test_story_tool_returns_valid_original_writing_brief():
    tool = build_generate_story_tool()
    args = GenerateStoryInput(
        genre="science fiction", theme="found family", target_words=250,
        main_characters=["Mira", "a repair robot"], tone="hopeful",
        constraints=["Use a lunar setting", "End with a discovery"],
    )
    result = await tool.handler(args)
    assert result["target_words"] == 250
    assert "Use a lunar setting" in result["drafting_brief"]
    assert "new prose" in result["originality_note"]
    assert tool.permission_policy == PermissionPolicy.AUTO

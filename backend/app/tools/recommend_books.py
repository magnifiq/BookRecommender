from typing import Literal

from pydantic import BaseModel, Field

from app.books.openlibrary import BookProviderUnavailable, OpenLibraryClient
from app.recommendations.ranker import RecommendationPreferences, RecommendationRanker
from app.tools.base import ToolDefinition


class RecommendBooksInput(RecommendationPreferences):
    limit: int = Field(default=5, ge=1, le=10)


class RecommendationToolResult(BaseModel):
    status: Literal["ok", "no_matches", "insufficient_preferences", "unavailable"]
    recommendations: list[dict] = Field(default_factory=list)
    message: str | None = None


def build_recommend_books_tool(
    provider: OpenLibraryClient,
    ranker: RecommendationRanker | None = None,
) -> ToolDefinition:
    recommendation_ranker = ranker or RecommendationRanker()

    async def recommend_books(arguments: RecommendBooksInput) -> dict:
        preferences = RecommendationPreferences.model_validate(
            arguments.model_dump(exclude={"limit"})
        )
        if not any((preferences.genres, preferences.favorite_authors,
                    preferences.topics, preferences.preferred_length, preferences.mood)):
            result = RecommendationToolResult(
                status="insufficient_preferences",
                message="Share a genre, author, topic, length, or mood to get grounded recommendations.",
            )
            return result.model_dump(mode="json")
        query_terms = [*preferences.genres, *preferences.favorite_authors, *preferences.topics]
        if preferences.mood:
            query_terms.append(preferences.mood)
        query = " ".join(query_terms)
        if not query:
            query = preferences.preferred_length or "books"
        try:
            books = await provider.search(query, search_by="any", limit=20)
        except BookProviderUnavailable:
            return RecommendationToolResult(
                status="unavailable",
                message="The book catalog is temporarily unavailable. No recommendations were invented.",
            ).model_dump(mode="json")
        ranked = recommendation_ranker.rank(preferences, books, arguments.limit)
        return RecommendationToolResult(
            status="ok" if ranked else "no_matches",
            recommendations=[item.model_dump(mode="json") for item in ranked],
            message=None if ranked else "No catalog records matched these preferences.",
        ).model_dump(mode="json")

    return ToolDefinition(
        name="recommend_books",
        description=(
            "Search Open Library and rank real catalog records against genres, "
            "authors, topics, length, mood, and exclusions. Never invent titles."
        ),
        args_schema=RecommendBooksInput,
        handler=recommend_books,
    )

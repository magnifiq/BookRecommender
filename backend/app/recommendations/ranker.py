from pydantic import BaseModel, Field
from typing import Literal

from app.books.models import BookRecord


class RecommendationPreferences(BaseModel):
    genres: list[str] = Field(default_factory=list, max_length=5)
    favorite_authors: list[str] = Field(default_factory=list, max_length=5)
    topics: list[str] = Field(default_factory=list, max_length=8)
    preferred_length: Literal["short", "medium", "long"] | None = None
    mood: str | None = Field(default=None, max_length=100)
    already_read_or_disliked: list[str] = Field(default_factory=list, max_length=20)


class RankedBook(BaseModel):
    book: BookRecord
    score: int
    reasons: list[str]


class RecommendationRanker:
    """Small, explainable metadata matcher; it does not infer quality or taste."""

    def rank(self, preferences: RecommendationPreferences,
             books: list[BookRecord], limit: int = 5) -> list[RankedBook]:
        excluded = {self._key(value) for value in preferences.already_read_or_disliked}
        ranked = []
        for book in books:
            book_keys = {self._key(book.title), *(self._key(author) for author in book.authors)}
            if book_keys & excluded:
                continue
            haystack = " ".join([book.title, *book.authors, *book.subjects, book.description or ""]).casefold()
            reasons: list[str] = []
            score = 0
            for genre in preferences.genres:
                if genre.casefold() in haystack:
                    score += 3
                    reasons.append(f"matches genre: {genre}")
            for topic in preferences.topics:
                if topic.casefold() in haystack:
                    score += 2
                    reasons.append(f"matches topic: {topic}")
            for author in preferences.favorite_authors:
                if any(author.casefold() == existing.casefold() for existing in book.authors):
                    score += 5
                    reasons.append(f"by preferred author: {author}")
            if preferences.mood and preferences.mood.casefold() in haystack:
                score += 1
                reasons.append(f"metadata mentions requested mood: {preferences.mood}")
            if preferences.preferred_length and book.page_count:
                length = preferences.preferred_length
                matches = (
                    length == "short" and book.page_count <= 250
                    or length == "medium" and 250 < book.page_count <= 450
                    or length == "long" and book.page_count > 450
                )
                if matches:
                    score += 1
                    reasons.append(f"fits preferred length: {length}")
            if score == 0:
                continue
            ranked.append(RankedBook(book=book, score=score, reasons=reasons))
        ranked.sort(key=lambda result: (-result.score, result.book.title.casefold()))
        return ranked[:limit]

    @staticmethod
    def _key(value: str) -> str:
        return " ".join(value.casefold().split())

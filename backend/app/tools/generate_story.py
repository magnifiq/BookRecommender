from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.tools.base import ToolDefinition


class GenerateStoryInput(BaseModel):
    genre: str = Field(min_length=1, max_length=80)
    theme: str = Field(min_length=1, max_length=160)
    target_words: int = Field(default=500, ge=50, le=1500)
    main_characters: list[str] = Field(default_factory=list, max_length=6)
    tone: str = Field(default="warm", max_length=80)
    constraints: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("genre", "theme", "tone")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must contain non-whitespace text")
        return value.strip()


class StoryBrief(BaseModel):
    status: Literal["ready"] = "ready"
    genre: str
    theme: str
    target_words: int
    main_characters: list[str]
    tone: str
    constraints: list[str]
    drafting_brief: str
    originality_note: str = "Write new prose; do not imitate or reproduce an existing book or living author's style."


def build_generate_story_tool() -> ToolDefinition:
    async def generate_story(arguments: GenerateStoryInput) -> dict:
        characters = arguments.main_characters or ["an original protagonist"]
        constraint_text = "\n".join(f"- {item}" for item in arguments.constraints) or "- No additional constraints."
        brief = (
            f"Write an original {arguments.genre} story about {arguments.theme}. "
            f"Target approximately {arguments.target_words} words, use a {arguments.tone} tone, "
            f"and feature: {', '.join(characters)}.\n"
            f"Additional constraints:\n{constraint_text}\n"
            "Use the supplied ingredients, but invent the plot, setting details, and prose. "
            "Return the story itself in the final response."
        )
        return StoryBrief(
            genre=arguments.genre,
            theme=arguments.theme,
            target_words=arguments.target_words,
            main_characters=characters,
            tone=arguments.tone,
            constraints=arguments.constraints,
            drafting_brief=brief,
        ).model_dump(mode="json")

    return ToolDefinition(
        name="generate_story",
        description=(
            "Prepare a structured brief for an original short story. After this "
            "tool returns, write the story yourself in the final response; never "
            "reproduce book text or imitate a living author's style."
        ),
        args_schema=GenerateStoryInput,
        handler=generate_story,
    )

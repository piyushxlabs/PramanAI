"""Tool schemas in dual Pydantic V2 and Strict JSON Schema formats."""
from src.tools.schemas.compare_go_versions import (
    COMPARE_GO_VERSIONS_JSON_SCHEMA,
    CompareGoVersionsInput,
    CompareGoVersionsOutput,
    SupersessionLink,
)
from src.tools.schemas.get_source_highlight import (
    GET_SOURCE_HIGHLIGHT_JSON_SCHEMA,
    BoundingBox,
    GetSourceHighlightInput,
    GetSourceHighlightOutput,
)
from src.tools.schemas.search_go_corpus import (
    SEARCH_GO_CORPUS_JSON_SCHEMA,
    PassageMatch,
    SearchGoCorpusInput,
    SearchGoCorpusOutput,
)

__all__ = [
    "SEARCH_GO_CORPUS_JSON_SCHEMA",
    "SearchGoCorpusInput",
    "SearchGoCorpusOutput",
    "PassageMatch",
    "COMPARE_GO_VERSIONS_JSON_SCHEMA",
    "CompareGoVersionsInput",
    "CompareGoVersionsOutput",
    "SupersessionLink",
    "GET_SOURCE_HIGHLIGHT_JSON_SCHEMA",
    "GetSourceHighlightInput",
    "GetSourceHighlightOutput",
    "BoundingBox",
]

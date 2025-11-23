from typing import Any
import numpy as np
from pydantic import BaseModel


class ParagraphEmbeddings(BaseModel):
    article_id: int
    article_title: str
    article_category: str
    paragraph: str
    embedding: np.ndarray

    class Config:
        arbitrary_types_allowed = True


class ParagraphSimilarities(BaseModel):
    article_id: int
    article_title: str
    article_category: str
    paragraph: str
    similarity: float


class Article(BaseModel):
    title: str
    content: str
    category: str


class SemanticSearchEngineResponse(BaseModel):
    categoria_mais_proxima: tuple[str, Any] | None
    top_categorias: list[tuple[str, Any]]
    top_paragrafos: list[ParagraphSimilarities]
    artigos_recomendados: list[Article]

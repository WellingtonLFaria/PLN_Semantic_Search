import numpy as np
from pydantic import BaseModel


class ParagraphEmbeddings(BaseModel):
    article_id: int
    article_title: str
    article_category: str
    paragraph: str
    embedding: np.ndarray


class ParagraphSimilarities(BaseModel):
    article_id: int
    article_title: str
    article_category: str
    paragraph: str
    similarity: float

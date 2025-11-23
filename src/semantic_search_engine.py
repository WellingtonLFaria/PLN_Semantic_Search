import json
import re

import nltk  # type: ignore[import-untyped]
import numpy as np
from gensim.models import Word2Vec  # type: ignore[import-untyped]
from nltk.corpus import stopwords  # type: ignore[import-untyped]
from nltk.tokenize import word_tokenize  # type: ignore[import-untyped]
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-untyped]

from src.config import settings
from src.schemas.schemas import (
    Article,
    ParagraphEmbeddings,
    ParagraphSimilarities,
    SemanticSearchEngineResponse,
)


def ensure_nltk_resources() -> None:
    resources = ["punkt", "stopwords", "punkt_tab"]
    for r in resources:
        try:
            if r == "stopwords":
                nltk.data.find(f"corpora/{r}")
            else:
                nltk.data.find(f"tokenizers/{r}")
        except LookupError:
            nltk.download(r)


ensure_nltk_resources()


class SemanticSearchEngine:
    def __init__(self, debug: bool = False) -> None:
        self.model: Word2Vec | None = None
        self.articles: list[Article] | None = None
        self.category_descriptions: dict[str, str] = {}
        self.category_embeddings: dict[str, np.ndarray] = {}
        self.paragraphs_embeddings: list[ParagraphEmbeddings] = []
        self.debug = debug

        self.stop_words_pt = set(stopwords.words("portuguese"))
        self.stop_words_en = set(stopwords.words("english"))
        self.stop_words = self.stop_words_pt.union(self.stop_words_en)

    def preprocess_text(self, text: str) -> list[str]:
        text = text.lower()

        text = re.sub(r"[^\w\sáàâãéèêíïóôõöúçñ\-_]", " ", text)

        text = re.sub(r"\b\d+\b", " ", text)

        tokens = word_tokenize(text)

        tokens = [
            token
            for token in tokens
            if (token not in self.stop_words and len(token) > 2)
            or (len(token) >= 2 and token.isupper())
        ]
        return tokens

    def build_category_descriptions(self) -> None:
        if not self.articles:
            return

        self.category_descriptions = {}

        for article in self.articles:
            category = article.category
            if category not in self.category_descriptions:
                self.category_descriptions[category] = []  # type: ignore[assignment]

            self.category_descriptions[category].append(article.title)  # type: ignore[attr-defined]

            first_sentences = " ".join(article.content.split(".")[:2])
            self.category_descriptions[category].append(first_sentences)  # type: ignore[attr-defined]

        for category in self.category_descriptions:
            titles = [
                desc
                for desc in self.category_descriptions[category]
                if desc == self.get_article_titles_by_category(category)[0]
            ]
            content_snippets = [
                desc
                for desc in self.category_descriptions[category]
                if desc not in titles
            ]

            final_description = " ".join(titles * 3 + content_snippets)
            self.category_descriptions[category] = final_description

    def get_article_titles_by_category(self, category: str) -> list[str]:
        if not self.articles:
            return []
        return [
            article.title for article in self.articles if article.category == category
        ]

    def train_model(
        self,
        texts: list[str],
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
    ) -> None:
        sentences = [self.preprocess_text(text) for text in texts]
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            sg=1,
        )

    def text_to_vector(self, text: str) -> np.ndarray:
        if self.model is None:
            return np.zeros(100)
        tokens = self.preprocess_text(text)
        if not tokens:
            return np.zeros(self.model.vector_size)
        vectors = []
        for token in tokens:
            if token in self.model.wv:
                vectors.append(self.model.wv[token])
        if vectors:
            return np.mean(vectors, axis=0)  # type: ignore[no-any-return]
        else:
            return np.zeros(self.model.vector_size)

    def calculate_category_similarity(self, query: str, category: str) -> float:
        if category not in self.category_embeddings:
            return 0.0
        query_embedding = self.text_to_vector(query)
        category_embedding = self.category_embeddings[category]
        similarity = cosine_similarity([query_embedding], [category_embedding])[0][0]
        if self.debug:
            print(f"   📐 {category}: {similarity:.3f}")
        return similarity  # type: ignore[no-any-return]

    def load_data(self) -> None:
        articles: list[Article] = []
        with open(settings.articles_file_path, "r", encoding="utf-8") as file:
            content = file.read()
            for article_data in json.loads(content):
                articles.append(Article(**article_data))
        self.articles = articles
        if self.debug:
            print(f"📚 ARTIGOS CARREGADOS: {len(articles)}")
        contents = [article.content for article in articles]
        self.train_model(contents, vector_size=100, min_count=2)
        self.build_category_descriptions()
        for category, description in self.category_descriptions.items():
            self.category_embeddings[category] = self.text_to_vector(description)
        if self.debug:
            print("\n🔍 DESCRIÇÕES DAS CATEGORIAS (amostra):")
            for category, desc in list(self.category_descriptions.items())[:5]:
                print(f"   {category}: {desc[:100]}...")
        self.paragraphs_embeddings = []
        for article in articles:
            paragraphs = [p.strip() for p in article.content.split("\n") if p.strip()]
            for paragraph in paragraphs:
                embedding = self.text_to_vector(paragraph)
                self.paragraphs_embeddings.append(
                    ParagraphEmbeddings(
                        article_id=len(self.paragraphs_embeddings) + 1,
                        article_title=article.title,
                        article_category=article.category,
                        paragraph=paragraph,
                        embedding=embedding,
                    )
                )

    def search_semantic(
        self, query: str, top_k: int = 5
    ) -> SemanticSearchEngineResponse:
        if self.debug:
            print(f"\n🎯 INICIANDO BUSCA: '{query}'")
            print("📐 CALCULANDO SIMILARIDADES COM CATEGORIAS:")

        categories_similarities = {}
        for category in self.category_embeddings:
            similarity = self.calculate_category_similarity(query, category)
            categories_similarities[category] = similarity

        if self.debug:
            print("\n📊 TOP 10 CATEGORIAS MAIS SIMILARES:")
            for category, similarity in sorted(
                categories_similarities.items(), key=lambda x: x[1], reverse=True
            )[:10]:
                print(f"   {category}: {similarity:.3f}")

        query_embedding = self.text_to_vector(query)
        paragraphs_similarities: list[ParagraphSimilarities] = []
        for para_embedding in self.paragraphs_embeddings:
            similarity = cosine_similarity(
                [query_embedding], [para_embedding.embedding]
            )[0][0]
            paragraphs_similarities.append(
                ParagraphSimilarities(
                    article_id=para_embedding.article_id,
                    article_title=para_embedding.article_title,
                    article_category=para_embedding.article_category,
                    paragraph=para_embedding.paragraph,
                    similarity=similarity,
                )
            )

        sorted_categories = sorted(
            categories_similarities.items(), key=lambda x: x[1], reverse=True
        )
        sorted_paragraphs = sorted(
            paragraphs_similarities, key=lambda x: x.similarity, reverse=True
        )

        top_category = sorted_categories[0] if sorted_categories else None

        if self.debug and top_category:
            print(
                f"\n🏆 CATEGORIA SELECIONADA: {top_category[0]} (similaridade: {top_category[1]:.3f})"
            )

        return SemanticSearchEngineResponse(
            categoria_mais_proxima=top_category,
            top_categorias=sorted_categories[:3],
            top_paragrafos=sorted_paragraphs[:top_k],
            artigos_recomendados=self._filter_articles_by_category(
                top_category[0] if top_category else None
            ),
        )

    def _filter_articles_by_category(self, category: str | None) -> list[Article]:
        if category and self.articles:
            return [
                article for article in self.articles if article.category == category
            ]
        return []


def demonstrar_sistema() -> None:
    sistema = SemanticSearchEngine(debug=True)
    sistema.load_data()

    consultas = [
        "redes neurais e inteligência artificial",
        "análise de textos e sentimentos",
        "modelos de representação de palavras",
        "processamento de linguagem natural",
        "aprendizado de máquina",
    ]

    for consulta in consultas:
        print(f"\n{'=' * 60}")
        print(f"🔍 BUSCA: '{consulta}'")
        print(f"{'=' * 60}")

        resultados = sistema.search_semantic(consulta, top_k=3)

        if resultados.categoria_mais_proxima:
            cat_nome, cat_similaridade = resultados.categoria_mais_proxima
            print(
                f"📂 CATEGORIA MAIS PRÓXIMA: {cat_nome} (similaridade: {cat_similaridade:.3f})"
            )
        else:
            print("📂 NENHUMA CATEGORIA RELEVANTE ENCONTRADA")

        print(f"\n📝 TOP {len(resultados.top_paragrafos)} PARÁGRAFOS RELACIONADOS:")
        for i, para in enumerate(resultados.top_paragrafos, 1):
            print(f"   {i}. [{para.similarity:.3f}] {para.paragraph.strip()[:100]}...")
            print(
                f"      📖 Artigo: {para.article_title} | Categoria: {para.article_category}"
            )

        if resultados.artigos_recomendados:
            if resultados.categoria_mais_proxima:
                print(
                    f"\n📚 ARTIGOS RECOMENDADOS NA CATEGORIA '{resultados.categoria_mais_proxima[0]}':"
                )
            for artigo in resultados.artigos_recomendados:
                print(f"   - {artigo.title}")

        print(f"\n{'=' * 60}")


if __name__ == "__main__":
    demonstrar_sistema()

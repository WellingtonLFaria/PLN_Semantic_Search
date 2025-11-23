import re

import nltk
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

from src.schemas.schemas import ParagraphEmbeddings, ParagraphSimilarities

nltk.download("punkt")
nltk.download("stopwords")


class SemanticSearchEngine:
    def __init__(self):
        self.model = None
        self.model_doc2vec = None
        self.data = None
        self.categories_embeddings = {}
        self.paragraphs_embeddings = []
        self.stop_words = set(stopwords.words("portuguese"))
        self.stemmer = PorterStemmer()

    def preprocess_text(self, text: str):
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\d+", " ", text)
        tokens = word_tokenize(text, language="portuguese")
        tokens = [
            self.stemmer.stem(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        return tokens

    def train_model(self, texts: list[str], vector_size=100, window=5, min_count=1):
        sentences = [self.preprocess_text(texto) for texto in texts]
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
        )
        tagged_docs = [
            TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(sentences)
        ]
        self.model_doc2vec = Doc2Vec(
            documents=tagged_docs,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
        )

    def text_to_vector(self, text: str) -> np.ndarray:
        tokens = self.preprocess_text(text)
        vectors = []
        for token in tokens:
            if token in self.model.wv:
                vectors.append(self.model.wv[token])
        if len(vectors) > 0:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.model.vector_size)

    def load_data(self, data: pd.DataFrame):
        self.data = data
        texts = []
        for _, article in data.iterrows():
            texts.append(article["categoria"])
            texts.append(article["conteudo"])
        self.train_model(texts)
        unique_categories = data["categoria"].unique()
        for category in unique_categories:
            self.categories_embeddings[category] = self.text_to_vector(category)

        self.paragraphs_embeddings = []
        for idx, article in data.iterrows():
            paragraphs = article["conteudo"].split("\n")
            for _, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    embedding = self.text_to_vector(paragraph)
                    self.paragraphs_embeddings.append(
                        ParagraphEmbeddings(
                            article_id=idx,
                            article_title=article["titulo"],
                            article_category=article["categoria"],
                            paragraph=paragraph,
                            embedding=embedding,
                        )
                    )

    def search_semantic(self, query: str, top_k=5):
        consulta_embedding = self.text_to_vector(query)

        categories_simalarities = {}
        for category, cat_embedding in self.categories_embeddings.items():
            similarity = cosine_similarity([consulta_embedding], [cat_embedding])[0][0]
            categories_simalarities[category] = similarity

        paragraphs_similarities = []
        for para_info in self.paragraphs_embeddings:
            similarity = cosine_similarity(
                [consulta_embedding], [para_info["embedding"]]
            )[0][0]
            paragraphs_similarities.append(
                ParagraphSimilarities(
                    article_id=para_info["artigo_id"],
                    article_title=para_info["artigo_titulo"],
                    article_category=para_info["artigo_categoria"],
                    paragraph=para_info["paragrafo"],
                    similarity=similarity,
                )
            )

        sorted_categories = sorted(
            categories_simalarities.items(), key=lambda x: x[1], reverse=True
        )
        sorted_paragraphs = sorted(
            paragraphs_similarities, key=lambda x: x["similaridade"], reverse=True
        )

        return {
            "categoria_mais_proxima": sorted_categories[0]
            if sorted_categories
            else None,
            "top_categorias": sorted_categories[:3],
            "top_paragrafos": sorted_paragraphs[:top_k],
            "artigos_recomendados": self._filter_article_per_category(
                sorted_categories[0][0] if sorted_categories else None
            ),
        }

    def _filter_article_per_category(self, category):
        if category and self.data is not None:
            return self.data[self.data["categoria"] == category].to_dict("records")
        return []


def criar_dados_exemplo():
    dados = pd.DataFrame(
        {
            "titulo": [
                "Introdução ao Processamento de Linguagem Natural",
                "Redes Neurais Artificiais Aplicadas ao PLN",
                "Análise de Sentimentos com Deep Learning",
                "Modelos de Linguagem e sua Evolução",
            ],
            "categoria": [
                "PLN",
                "PLN",
                "Machine Learning",
                "Linguística Computacional",
            ],
            "conteudo": [
                """O Processamento de Linguagem Natural é uma área da inteligência artificial que estuda a interação entre computadores e linguagem humana.
            As técnicas de PLN permitem que máquinas entendam, interpretem e gerem linguagem natural.
            Word2Vec é um algoritmo popular para criar representações vetoriais de palavras.""",
                """Redes neurais artificiais revolucionaram o campo do PLN.
            Modelos como Word2Vec e BERT utilizam arquiteturas neurais profundas.
            A representação distribuída de palavras captura relações semânticas complexas.""",
                """Análise de sentimentos é uma aplicação prática do machine learning.
            Usamos redes neurais para classificar textos como positivos, negativos ou neutros.
            Word embeddings melhoram significativamente o desempenho desses sistemas.""",
                """Modelos de linguagem evoluíram de abordagens estatísticas para neurais.
            A representação semântica é crucial para entender o significado do texto.
            Técnicas modernas capturam contexto e polissemia nas palavras.""",
            ],
        }
    )
    return dados


def demonstrar_sistema():
    sistema = SemanticSearchEngine()

    dados = criar_dados_exemplo()
    sistema.load_data(dados)

    consultas = [
        "redes neurais e inteligência artificial",
        "análise de textos e sentimentos",
        "modelos de representação de palavras",
        "processamento de linguagem natural",
    ]

    for consulta in consultas:
        print(f"\n=== Busca: '{consulta}' ===")
        resultados = sistema.search_semantic(consulta)

        print(f"Categoria mais próxima: {resultados['categoria_mais_proxima']}")

        print("\nTop 3 parágrafos relacionados:")
        for i, para in enumerate(resultados["top_paragrafos"]):
            print(
                f"{i + 1}. [相似度: {para['similaridade']:.3f}] {para['paragrafo'][:100]}..."
            )

        print(f"\nArtigos na categoria '{resultados['categoria_mais_proxima'][0]}':")
        for artigo in resultados["artigos_recomendados"]:
            print(f"- {artigo['titulo']}")


if __name__ == "__main__":
    demonstrar_sistema()

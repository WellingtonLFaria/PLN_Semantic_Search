import json
import re

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

from src.config import settings
from src.schemas.schemas import (
    Article,
    ParagraphEmbeddings,
    ParagraphSimilarities,
    SemanticSearchEngineResponse,
)


def ensure_nltk_resources() -> None:
    resources = ["punkt", "stopwords"]
    for r in resources:
        try:
            if r == "stopwords":
                nltk.data.find(f"corpora/{r}")
            else:
                nltk.data.find(f"tokenizers/{r}")
        except LookupError:
            nltk.download(r)


ensure_nltk_resources()


class SemanticSearchEngineTransformers:
    def __init__(self, debug: bool = False) -> None:
        self.model = None
        self.tokenizer = None
        self.articles: list[Article] | None = None
        self.category_embeddings: dict[str, np.ndarray] = {}
        self.paragraphs_embeddings: list[ParagraphEmbeddings] = []
        self.debug = debug

        # Stopwords em português e inglês
        self.stop_words_pt = set(stopwords.words("portuguese"))
        self.stop_words_en = set(stopwords.words("english"))
        self.stop_words = self.stop_words_pt.union(self.stop_words_en)

        # Carregar modelo transformers
        self._load_model()

    def _load_model(self):
        """Carrega o modelo e tokenizer do transformers"""
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        if self.debug:
            print(f"🔄 Carregando modelo: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            if self.debug:
                print("✅ Modelo carregado na GPU")
        else:
            if self.debug:
                print("✅ Modelo carregado na CPU")

    def preprocess_text(self, text: str) -> str:
        """Pré-processamento simples para limpeza do texto"""
        text = text.lower()
        # Limpeza básica - mantém caracteres relevantes
        text = re.sub(r"[^\w\sáàâãéèêíïóôõöúçñ\-_.,!?;:]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def get_embedding(self, text: str) -> np.ndarray:
        """Gera embedding usando modelo transformers"""
        if not text.strip():
            return np.zeros(384)  # Dimensão do modelo MiniLM-L12

        text = self.preprocess_text(text)

        # Tokenizar e criar tensores
        inputs = self.tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )

        # Mover para GPU se disponível
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Gerar embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Usar mean pooling na última camada hidden states
        embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        # Aplicar mean pooling com attention mask
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        )
        sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask

        # Converter para numpy e normalizar
        embedding = embedding.cpu().numpy()
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

        return embedding[0]

    def build_category_representations(self) -> None:
        """Constrói representações para cada categoria baseado nos artigos"""
        if not self.articles:
            return

        category_texts = {}

        for article in self.articles:
            category = article.category
            if category not in category_texts:
                category_texts[category] = []

            # Usar título e primeiras sentenças do conteúdo
            title = article.title
            first_sentences = ". ".join(sent_tokenize(article.content)[:2]) + "."

            category_texts[category].extend([title, first_sentences])

        # Gerar embedding para cada categoria
        for category, texts in category_texts.items():
            # Combinar todos os textos da categoria
            combined_text = " ".join(texts)
            self.category_embeddings[category] = self.get_embedding(combined_text)

            if self.debug:
                print(f"   📊 Embedding da categoria '{category}' gerado")

    def load_data(self) -> None:
        """Carrega e processa os dados"""
        articles: list[Article] = []
        with open(settings.articles_file_path, "r", encoding="utf-8") as file:
            content = file.read()
            for article_data in json.loads(content):
                articles.append(Article(**article_data))

        self.articles = articles

        if self.debug:
            print(f"📚 ARTIGOS CARREGADOS: {len(articles)}")

        # Constrói representações das categorias
        if self.debug:
            print("🔄 Construindo representações das categorias...")
        self.build_category_representations()

        # Prepara embeddings dos parágrafos
        if self.debug:
            print("🔄 Gerando embeddings dos parágrafos...")
        self.paragraphs_embeddings = []

        for i, article in enumerate(articles):
            if self.debug and i % 10 == 0:
                print(f"   📝 Processando artigo {i + 1}/{len(articles)}")

            paragraphs = [p.strip() for p in article.content.split("\n") if p.strip()]
            for paragraph in paragraphs:
                embedding = self.get_embedding(paragraph)
                self.paragraphs_embeddings.append(
                    ParagraphEmbeddings(
                        article_id=len(self.paragraphs_embeddings) + 1,
                        article_title=article.title,
                        article_category=article.category,
                        paragraph=paragraph,
                        embedding=embedding,
                    )
                )

        if self.debug:
            print(f"✅ {len(self.paragraphs_embeddings)} parágrafos processados")

    def calculate_category_similarity(self, query: str, category: str) -> float:
        """Calcula similaridade entre consulta e categoria"""
        if category not in self.category_embeddings:
            return 0.0

        query_embedding = self.get_embedding(query)
        category_embedding = self.category_embeddings[category]

        similarity = cosine_similarity([query_embedding], [category_embedding])[0][0]

        if self.debug:
            print(f"   📐 {category}: {similarity:.3f}")

        return similarity

    def search_semantic(
        self, query: str, top_k: int = 5
    ) -> SemanticSearchEngineResponse:
        """Busca semântica usando transformers"""
        if self.debug:
            print(f"\n🎯 INICIANDO BUSCA: '{query}'")
            print("📐 CALCULANDO SIMILARIDADES COM CATEGORIAS:")

        # Calcula similaridade com categorias
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

        # Calcula similaridade com parágrafos
        query_embedding = self.get_embedding(query)
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

        # Ordena resultados
        sorted_categories = sorted(
            categories_similarities.items(), key=lambda x: x[1], reverse=True
        )
        sorted_paragraphs = sorted(
            paragraphs_similarities, key=lambda x: x.similarity, reverse=True
        )

        # Seleciona categoria mais similar
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
        """Filtra artigos por categoria"""
        if category and self.articles:
            return [
                article for article in self.articles if article.category == category
            ]
        return []


def demonstrar_sistema_transformers() -> None:
    """Função de demonstração do sistema com transformers"""
    sistema = SemanticSearchEngineTransformers(debug=True)

    if sistema.debug:
        print("🚀 INICIANDO SISTEMA COM TRANSFORMERS")

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
    demonstrar_sistema_transformers()

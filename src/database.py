import json

import numpy as np
from pydantic import BaseModel
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func

from config import settings

Base = declarative_base()


class Category(Base):  # type: ignore[misc, valid-type]
    __tablename__ = "category"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), unique=True, nullable=False)
    embedding_vector = Column(Text, nullable=True)
    documents = relationship("Document", back_populates="category")


class Document(Base):  # type: ignore[misc, valid-type]
    __tablename__ = "document"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(500), nullable=False)
    category_id = Column(Integer, ForeignKey("category.id"), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=func.now())

    category = relationship("Category", back_populates="documents")
    paragraphs = relationship("Paragraph", back_populates="document")


class Paragraph(Base):  # type: ignore[misc, valid-type]
    __tablename__ = "paragraph"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("document.id"), nullable=False)
    paragraph_text = Column(Text, nullable=False)
    embedding_vector = Column(Text, nullable=True)
    similarity_category = Column(Float, nullable=True)
    created_at = Column(DateTime, default=func.now())

    document = relationship("Document", back_populates="paragraphs")


def initialize_db():
    engine = create_engine(settings, echo=True)
    Base.metadata.create_all(engine)
    return engine


def serialize_vector(vector):
    return json.dumps(vector.tolist())


def deserialize_vector(vector_str):
    return np.array(json.loads(vector_str))


class Article(BaseModel):
    category: str
    title: str
    content: str


class SistemaBuscaSemanticaDB:
    def __init__(self, session: Session):
        self.session = session
        self.model = None

    def inserir_artigos_batch(self, articles: list[Article]):
        for article in articles:
            category = (
                self.session.query(Category).filter_by(name=article.category).first()
            )
            if not category:
                category = Category(name=article.category)
                self.session.add(category)
            document = Document(
                title=article.title, category=category, content=article.content
            )
            self.session.add(document)
        self.session.commit()

    def gerar_e_armazenar_embeddings(self):
        articles = self.session.query(Document).all()

        unique_categories = set([article.category.name for article in articles])
        for category_name in unique_categories:
            embedding = self.text_to_vector(category_name)
            category = (
                self.session.query(Category).filter_by(name=category_name).first()
            )
            category.embedding_vector = serialize_vector(embedding)

        for article in articles:
            paragraphs = article.content.split("\n")
            for _, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    embedding = self.text_to_vector(paragraph)
                    paragraph = Paragraph(
                        document_id=article.id,
                        paragraph_text=paragraph,
                        embedding_vector=serialize_vector(embedding),
                    )
                    self.session.add(paragraph)
        self.session.commit()

    def buscar_semantica_db(self, consulta, top_k=5):
        consulta_embedding = self.text_to_vector(consulta)

        categorias = self.session.query(Category).all()
        categoria_mais_proxima = None
        maior_similaridade_categoria = -1

        for categoria in categorias:
            categoria_embedding = deserialize_vector(categoria.embedding_vector)
            similaridade = np.dot(consulta_embedding, categoria_embedding)
            if similaridade > maior_similaridade_categoria:
                maior_similaridade_categoria = similaridade
                categoria_mais_proxima = categoria.name

        paragrafos_result = []
        paragrafos = self.session.query(Paragraph).all()
        for par in paragrafos:
            par_embedding = deserialize_vector(par.embedding_vector)
            similaridade_par = np.dot(consulta_embedding, par_embedding)
            paragrafos_result.append(
                {
                    "id": par.id,
                    "artigo_id": par.document_id,
                    "paragrafo": par.paragraph_text,
                    "similaridade": similaridade_par,
                }
            )

        paragrafos_result = sorted(
            paragrafos_result, key=lambda x: x["similaridade"], reverse=True
        )[:top_k]

        return {
            "categoria_mais_proxima": categoria_mais_proxima,
            "paragrafos_similares": paragrafos_result,
        }

    def text_to_vector(self, texto: str):
        tokens = self.preprocessar_texto(texto)
        vectors = []
        for token in tokens:
            if token in self.model.wv:
                vectors.append(self.model.wv[token])

        if len(vectors) > 0:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.model.vector_size)


def exemplo_uso_com_banco():
    engine = initialize_db()
    Session = sessionmaker(bind=engine)
    session = Session()

    sistema = SistemaBuscaSemanticaDB(session)

    artigos_exemplo = [
        Article(
            title="PLN com Redes Neurais",
            category="Processamento de Linguagem Natural",
            content="Redes neurais revolucionaram o PLN...\nWord2Vec é um algoritmo importante...",
        )
    ]
    sistema.inserir_artigos_batch(artigos_exemplo)
    sistema.gerar_e_armazenar_embeddings()
    resultados = sistema.buscar_semantica_db("redes neurais artificiais")
    print(resultados)


exemplo_uso_com_banco()

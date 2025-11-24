import json
from typing import Any, Dict

from flask import Flask, redirect, render_template, request, session, url_for
from werkzeug.wrappers.response import Response

from src.semantic_search_engine import SemanticSearchEngine
from src.semantic_search_engine_transformers import SemanticSearchEngineTransformers

search_engine_word2vec = SemanticSearchEngine(debug=False)
search_engine_transformers = SemanticSearchEngineTransformers(debug=False)


search_engine_word2vec.load_data()
search_engine_transformers.load_data()

app = Flask(__name__)
app.secret_key = "secret-key"


def get_current_engine() -> (
    tuple[SemanticSearchEngineTransformers, str] | tuple[SemanticSearchEngine, str]
):
    engine_type = session.get("engine_type", "word2vec")
    if engine_type == "transformers":
        return search_engine_transformers, "transformers"
    else:
        return search_engine_word2vec, "word2vec"


def format_search_results(resultados: Any) -> Dict[str, Any]:
    if not resultados:
        return {}

    formatted: dict[str, Any] = {
        "categoria_mais_proxima": None,
        "top_paragrafos": [],
        "artigos_recomendados": [],
        "engine_type": session.get("engine_type", "word2vec"),
    }

    if resultados.categoria_mais_proxima:
        cat_nome, cat_similaridade = resultados.categoria_mais_proxima
        formatted["categoria_mais_proxima"] = {
            "nome": cat_nome,
            "similaridade": f"{cat_similaridade:.3f}",
        }

    for i, para in enumerate(resultados.top_paragrafos[:5], 1):
        formatted["top_paragrafos"].append(
            {
                "numero": i,
                "similaridade": f"{para.similarity:.3f}",
                "texto": para.paragraph.strip()[:150] + "..."
                if len(para.paragraph.strip()) > 150
                else para.paragraph.strip(),
                "titulo_artigo": para.article_title,
                "categoria_artigo": para.article_category,
            }
        )

    for artigo in resultados.artigos_recomendados:
        formatted["artigos_recomendados"].append(
            {"titulo": artigo.title, "categoria": artigo.category}
        )

    return formatted


@app.route("/", methods=["GET"])
def index() -> str:
    result_data = request.args.get("result_data")
    query = request.args.get("query", "")

    current_engine, _ = get_current_engine()
    engine_type = (
        "transformers" if current_engine == search_engine_transformers else "word2vec"
    )

    result_dict = {}
    if result_data:
        try:
            result_dict = json.loads(result_data)
        except json.JSONDecodeError:
            result_dict = {}

    if result_dict and "engine_type" not in result_dict:
        result_dict["engine_type"] = engine_type

    return render_template(
        "index.html", result=result_dict, query=query, engine_type=engine_type
    )


@app.route("/search", methods=["POST"])
def semantic_search() -> Any:
    user_input = str(request.form["user_input"]).strip()

    if not user_input:
        return redirect(url_for("index"))

    current_engine, _ = get_current_engine()
    resultados = current_engine.search_semantic(user_input, top_k=5)
    result_dict = format_search_results(resultados)
    result_json = json.dumps(result_dict)

    return redirect(url_for("index", result_data=result_json, query=user_input))


@app.route("/switch-engine", methods=["POST"])
def switch_engine() -> Response:
    current_engine = session.get("engine_type", "word2vec")

    if current_engine == "word2vec":
        session["engine_type"] = "transformers"
    else:
        session["engine_type"] = "word2vec"

    query = request.args.get("query", "")
    result_data = request.args.get("result_data", "")

    if query and result_data:
        return redirect(url_for("index", query=query, result_data=result_data))
    elif query:
        return redirect(url_for("index", query=query))
    else:
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)

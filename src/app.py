import json
from typing import Any, Dict

from flask import Flask, redirect, render_template, request, url_for

from src.semantic_search_engine import SemanticSearchEngine

search_engine = SemanticSearchEngine(debug=False)
search_engine.load_data()

app = Flask(__name__)


def format_search_results(resultados: Any) -> Dict[str, Any]:
    if not resultados:
        return {}

    formatted: dict[str, Any] = {
        "categoria_mais_proxima": None,
        "top_paragrafos": [],
        "artigos_recomendados": [],
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

    # Formatar artigos recomendados
    for artigo in resultados.artigos_recomendados:
        formatted["artigos_recomendados"].append(
            {"titulo": artigo.title, "categoria": artigo.category}
        )

    return formatted


@app.route("/", methods=["GET"])
def index() -> str:
    result_data = request.args.get("result_data")
    query = request.args.get("query", "")

    # Converter string JSON de volta para objeto se existir
    result_dict = {}
    if result_data:
        try:
            result_dict = json.loads(result_data)
        except json.JSONDecodeError:
            result_dict = {}

    return render_template("index.html", result=result_dict, query=query)


@app.route("/search", methods=["POST"])
def semantic_search() -> Any:
    user_input = str(request.form["user_input"]).strip()

    if not user_input:
        return redirect(url_for("index"))

    # Realizar busca semântica
    resultados = search_engine.search_semantic(user_input, top_k=5)

    # Formatar resultados
    result_dict = format_search_results(resultados)

    # Converter para JSON para passar via URL
    result_json = json.dumps(result_dict)

    return redirect(url_for("index", result_data=result_json, query=user_input))


if __name__ == "__main__":
    app.run(debug=True)

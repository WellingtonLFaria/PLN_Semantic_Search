from flask import Flask, redirect, render_template, request, url_for

from src.database import Base, initialize_db

Base.metadata.create_all(initialize_db())
app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    result = request.args.get("result")
    return render_template("index.html", result=result)


@app.route("/search", methods=["POST"])
def semantic_search():
    user_input = str(request.form["user_input"])
    result = f"Resultado para: {user_input}"
    return redirect(url_for("index", result=result))


if __name__ == "__main__":
    app.run()

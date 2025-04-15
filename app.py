from flask import Flask, request, render_template
from search_engine import ImageSearchEngine
import os

app = Flask(__name__)
engine = ImageSearchEngine("image_surrogates.txt")

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    query = ""
    if request.method == "POST":
        query = request.form.get("query")
        if query:
            results = engine.search(query, top_k=10)
    return render_template("index.html", results=results, query=query)

if __name__ == "__main__":
    app.run(debug=True)

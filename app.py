# by:Snowkingliu
# 2024/6/22 16:22
import flask
from flask import render_template, redirect

from llm import chat

app = flask.Flask(__name__)


@app.route("/input", methods=["POST"])
def input_form():
    req = flask.request.form
    question = req.get("text")
    if not question:
        return redirect("/")
    answer = chat("answer")
    return render_template(
        "index.html", history=[{"question": question, "answer": answer}]
    )


@app.route("/")
def index():
    return render_template(
        "index.html",
        history=[
            (
                "一共几辆出租车",
                "根据提供的上下文信息，代码返回了4，这表示在出租车的行程数据中，有4个不同的出租车ID。因此，回答问题的答案是：一共有4辆车。",
            )
        ],
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)

"""Inline backtest data into the app: src/app.html + bt-data.json -> index.html"""
import os
here = os.path.dirname(os.path.abspath(__file__))
html = open(os.path.join(here, "src", "app.html"), encoding="utf-8").read()
data = open(os.path.join(here, "bt-data.json"), encoding="utf-8").read()
assert "</script" not in data
open(os.path.join(here, "index.html"), "w", encoding="utf-8").write(html.replace("/*BTDATA*/", data))
print("index.html", len(html) + len(data), "bytes")

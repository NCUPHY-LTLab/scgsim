"""Stage shared assets and render the Traditional Chinese Quarto project.

The root project owns English; zh-TW owns its translated prose. Shared theme
and scientific assets are copied for rendering, following Askr's child-project
pattern. Neither a solver nor a notebook is executed.
"""
from pathlib import Path
import json
import os
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CHINESE = ROOT / "zh-TW"


def stage_assets():
    for relative in ("_extensions", "docs/assets", "docs/images"):
        source, destination = ROOT / relative, CHINESE / relative
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
    labels = {
        "Source layout": "來源版圖",
        "Materials and electrical Nets": "材料與電氣網路",
        "Physical question": "物理問題",
        "Model and solver settings": "模型與求解器設定",
        "Prepare solver inputs": "準備求解器輸入",
        "Geometry, mesh or native spec": "幾何、網格或原生規格",
        "Run locally": "本機執行",
        "Native solver": "原生求解器",
        "Transfer handoff": "傳送交接套件",
        "Run on solver host": "在求解主機執行",
        "Read results and reports": "讀取結果與報表",
        "Returned outputs and interpretation": "回收輸出與解讀",
    }
    workflow = CHINESE / "docs/assets/scgsim-workflow.svg"
    text = workflow.read_text()
    for original, translated in labels.items():
        text = text.replace(">" + original + "<", ">" + translated + "<")
    workflow.write_text(text)

    diagram_labels = {
        "cpw-source.svg": {
            "Declared finite-ground CPW source geometry": "公開有限接地 CPW 來源幾何",
            "A 500 micrometre long substrate rectangle carries a 10 micrometre center signal with 6 micrometre gaps and two 80 micrometre grounds. Modal integration lines are shown at the negative and positive X end faces. Geometry only, no solver results.": "長 500 微米的基板矩形上有寬 10 微米的中央訊號線，與兩條寬 80 微米的接地相隔 6 微米。圖中於負 X 與正 X 端面標出 Modal 積分線。此圖僅呈現幾何，不含求解器結果。",
            "Finite-ground CPW source · dimensions in µm": "有限接地 CPW 來源 · 尺寸單位：µm",
            "Si substrate · 500 × 182": "Si 基板 · 500 × 182",
            "Ground · 80": "接地 · 80",
            "6 µm gap": "6 µm 間隙",
            "Signal · 10": "訊號線 · 10",
            "Port 1 · −X": "埠 1 · −X",
            "Port 2 · +X": "埠 2 · +X",
            "500 µm · source length": "500 µm · 來源長度",
            "Modal integration-line": "Modal 積分線",
            "endpoints are declared": "端點已明確宣告",
        },
        "q2d-cross-section.svg": {
            "Explicit Q2D CPW cross-section": "明確的 Q2D CPW 橫截面",
            "Four declared rectangles: 200 by 500 micrometre silicon substrate, 10 by 0.2 micrometre signal, and two 40 by 0.2 micrometre grounds with 10 micrometre gaps. The vertical scale is enlarged for visibility. Geometry only, no solver results.": "四個明確宣告的矩形：200 × 500 微米矽基板、10 × 0.2 微米訊號線，以及兩個 40 × 0.2 微米接地，間隙為 10 微米。為方便檢視，垂直比例已放大。此圖僅呈現幾何，不含求解器結果。",
            "Q2D CPW · baseline declared rectangles": "Q2D CPW · 基準矩形宣告",
            "Cross-section geometry only · vertical scale enlarged for clarity": "僅呈現橫截面幾何 · 放大垂直比例以利檢視",
            "Si substrate": "Si 基板",
            "GroundLeft · 40": "左側接地 · 40",
            "GroundRight · 40": "右側接地 · 40",
            "Signal · 10": "訊號線 · 10",
            "10 µm gap": "10 µm 間隙",
            "Substrate width · 200 µm": "基板寬度 · 200 µm",
            "Metal thickness: 0.2 µm": "金屬厚度：0.2 µm",
            "displayed enlarged": "圖中已放大顯示",
        },
    }
    for name, labels in diagram_labels.items():
        diagram = CHINESE / "docs" / "assets" / name
        content = diagram.read_text()
        for original, translated in labels.items():
            content = content.replace(original, translated)
        diagram.write_text(content)


def main():
    if os.environ.get("QUARTO_PROJECT_RENDER_ALL") != "1":
        return
    stage_assets()
    child_render = ["quarto", "render", str(CHINESE)]
    if (CHINESE / "_quarto-pages.yml").is_file():
        child_render.extend(["--profile", "pages"])
    child_render.append("--no-execute")
    subprocess.run(child_render, cwd=ROOT, check=True)
    # Retain the earlier Preview's English SGB URL as a compatibility redirect.
    alias = ROOT / "_site/docs/concepts/geometry-en.html"
    alias.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                     '<meta http-equiv="refresh" content="0;url=geometry.html">'
                     '<title>SGB documentation</title></head><body>'
                     '<a href="geometry.html">Read the English SGB documentation</a>'
                     '</body></html>\n')
    source = ROOT / "_site"
    documents = ["index.html"] + sorted(
        str(path.relative_to(ROOT).with_suffix(".html"))
        for path in (ROOT / "docs").rglob("*.qmd")
        if path.name != "geometry-en.qmd"
    )
    for edition in (source, source / "zh-TW"):
        for document in documents:
            if not (edition / document).is_file():
                raise FileNotFoundError(edition / document)
    manifest = {
        "english": {"root": "/", "pages": documents},
        "zh-TW": {"root": "/zh-TW/", "pages": documents},
    }
    (source / "bilingual-pages.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

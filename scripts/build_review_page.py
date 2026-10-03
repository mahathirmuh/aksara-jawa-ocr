"""Fase 6 — bangun halaman verifikasi label sebagai SATU file HTML mandiri.

  python scripts/build_review_page.py                  # -> out/verifikasi_aksara.html

Citra potongan, font Noto Sans Javanese, transliterator honocoroko (MIT), dan nama
codepoint ditanam ke dalam file, sehingga rekan pembaca aksara cukup membukanya di
browser (tanpa internet, tanpa akun). Progres tersimpan di browser; hasilnya
diekspor sebagai TSV lalu diimpor dengan scripts/import_review.py.
"""

import argparse
import base64
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMONS = ROOT / "data" / "real" / "commons"
TEMPLATE = ROOT / "scripts" / "review_template.html"
FONT = ROOT / "fonts" / "NotoSansJavanese-Regular.ttf"
HONOCOROKO = ROOT / "scripts" / "translit" / "node_modules" / "@naandalist" / "honocoroko"
OUTPUT = ROOT / "out" / "verifikasi_aksara.html"
MODULES = ["mappings.js", "transliterator.js", "index.js"]


def honocoroko_bundle() -> str:
    """Bungkus build CommonJS honocoroko dengan registri modul mini -> window.Honocoroko."""
    parts = []
    for name in MODULES:
        code = (HONOCOROKO / "dist" / "cjs" / name).read_text(encoding="utf-8")
        parts.append(f'"./{name}": function (module, exports, require) {{\n{code}\n}}')
    license_text = (HONOCOROKO / "LICENSE").read_text(encoding="utf-8") if (HONOCOROKO / "LICENSE").exists() else "MIT"
    bundle = (
        "/* @naandalist/honocoroko 1.4.0 — " + license_text.splitlines()[0].strip() + " */\n"
        "(function () {\n  var defs = {\n" + ",\n".join(parts) + "\n  };\n"
        "  var cache = {};\n"
        "  function req(name) {\n"
        "    if (cache[name]) return cache[name].exports;\n"
        "    var module = { exports: {} };\n"
        "    cache[name] = module;\n"
        "    defs[name](module, module.exports, req);\n"
        "    return module.exports;\n"
        "  }\n"
        "  window.Honocoroko = req('./index.js');\n"
        "})();"
    )
    return bundle.replace("</script", "<\\/script")


def codepoint_names() -> dict:
    names = {"0020": {"short": "spasi", "mark": False}, "200C": {"short": "ZWNJ", "mark": False}}
    for cp in range(0xA980, 0xA9E0):
        ch = chr(cp)
        full = unicodedata.name(ch, "")
        if not full:
            continue
        short = full.replace("JAVANESE ", "")
        for prefix in ("LETTER ", "VOWEL SIGN ", "CONSONANT SIGN ", "SIGN ", "DIGIT "):
            short = short.replace(prefix, "")
        names[f"{cp:04X}"] = {"short": short.lower(), "mark": unicodedata.category(ch).startswith("M")}
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drafts", type=Path, default=COMMONS / "drafts.jsonl")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    raw = args.drafts.read_text(encoding="utf-8")
    drafts = [json.loads(line) for line in raw.splitlines()]
    rows = []
    for d in drafts:
        image = (args.drafts.parent / d["image"]).read_bytes()
        rows.append({
            "id": d["id"], "image": "data:image/png;base64," + base64.b64encode(image).decode("ascii"),
            "width": d["width"], "height": d["height"], "latin": d.get("latin", ""), "draft": d.get("draft", ""),
            "note": d.get("note", ""), "title": d.get("title", ""), "page": d.get("page", ""),
            "artist": d.get("artist", ""), "license": d.get("license", ""),
        })
    build_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    data = json.dumps({"build": build_id, "rows": rows, "names": codepoint_names()}, ensure_ascii=False)

    html = TEMPLATE.read_text(encoding="utf-8")
    for placeholder, value in (
        ("__FONT_BASE64__", base64.b64encode(FONT.read_bytes()).decode("ascii")),
        ("__DATA_JSON__", data.replace("</", "<\\/")),
        ("/*__HONOCOROKO_BUNDLE__*/", honocoroko_bundle()),
    ):
        if placeholder not in html:
            raise RuntimeError(f"penanda {placeholder} tidak ada di template")
        html = html.replace(placeholder, value, 1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8", newline="\n")
    print(f"{len(rows)} baris, build {build_id} -> {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")

    artifact = args.output.with_name(args.output.stem + "_artifact.html")
    artifact.write_text(strip_document_shell(html), encoding="utf-8", newline="\n")
    print(f"versi artifact -> {artifact}")


def strip_document_shell(html: str) -> str:
    """Versi untuk Artifact: platform memasang doctype/html/head/body dan meta sendiri.

    File lokal tetap memakai dokumen lengkap karena meta charset dibutuhkan saat
    dibuka langsung dari disk.
    """
    head_open = (
        '<!doctype html>\n<html lang="id">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    )
    body_open, tail = "</head>\n<body>\n", "</body>\n</html>\n"
    for marker in (head_open, body_open, tail):
        if html.count(marker) != 1:
            raise RuntimeError(f"penanda dokumen tidak ditemukan tepat sekali: {marker[:30]!r}")
    out = html.replace(head_open, "", 1).replace(body_open, "", 1)
    return out[: out.rindex(tail)]


if __name__ == "__main__":
    main()

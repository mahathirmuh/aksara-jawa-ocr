"""Test scripts/export_methods.py: kartu metode halaman Metode, pada repo palsu kecil dan (bila ada) data sebenarnya."""

import inspect
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))  # scripts/ bukan paket

import compare_runs  # noqa: E402
import export_methods as em  # noqa: E402
from export_results import ALPHA, BETA, LM_PATH, OFFICIAL_LINES, OFFICIAL_RUN, WIDTH  # noqa: E402
from src import charlm, train  # noqa: E402
from src.augment import PHYSICAL_ORDER  # noqa: E402
from src.dataset import MIN_FRAMES_PER_TARGET, RENDER_SIZE_RANGE, TRAIN_FONTS, H, to_tensor  # noqa: E402
from src.evaluate import TARGETS  # noqa: E402
from src.model import CRNN  # noqa: E402
from src.tokenizer import BLANK  # noqa: E402
from test_export_datasets import GA, KA, SECRET, fake_repo, save_ckpt, start, write_log  # noqa: E402  (repo palsu yang sama)

ROOT = Path(__file__).resolve().parents[1]
RUN = "fase7_track"  # nama run sungguhan, supaya kunci pipeline dan run kontrolnya dikenal
FAMILY_NOTE = ("Font uji javatext sekeluarga dengan font latih NotoSansJavanese-Regular dan Salinan (lebar 73 dari 73 "
               "aksara, angka, dan pada persis sama), jadi {} mengukur generalisasi di dalam keluarga huruf itu, bukan "
               "ke font yang belum pernah dilihat model.")


def test_number_formats_are_indonesian():
    assert em.num(4590909) == "4.590.909" and em.num(93) == "93"
    assert em.dec(0.5) == "0,5" and em.dec(0.0003, 6) == "0,0003" and em.dec(16.0) == "16" and em.dec(0.25) == "0,25"
    assert em.pct(0.2145905) == "21,46%" and em.pct(1.0, 0) == "100%" and em.pct(0.872, 1) == "87,2%"
    assert em.points(-0.0929) == "−9,29 poin" and em.points(0.05) == "+5,00 poin"
    assert em.interval(-0.1192, -0.0683) == "−11,92 sampai −6,83" and em.interval(-0.001, 0.002) == "−0,10 sampai +0,20"
    assert [em.listing(items) for items in ([], ["a"], ["a", "b"], ["a", "b", "c"])] == ["", "a", "a dan b", "a, b, dan c"]


def test_entry_rejects_unknown_kind_and_status():
    plain = em.entry("a", "A", "model", "official", "x")
    assert plain["settings"] == [] and plain["note"] is None and plain["evidence"] is None
    assert em.entry("a", "A", "model", "official", "x", evidence=None, evidence_source="berkas")["evidence_source"] is None
    assert em.entry("a", "A", "model", "official", "x", note="batasan")["note"] == "batasan"
    with pytest.raises(ValueError):
        em.entry("a", "A", "ajaib", "official", "x")
    with pytest.raises(ValueError):
        em.entry("a", "A", "model", "rahasia", "x")


@pytest.mark.parametrize("name", sorted(em.CODE_FACTS))
def test_statements_about_the_code_follow_the_source(monkeypatch, name):
    code = em.code_facts()
    assert code == {"optimizer": "AdamW", "scheduler": "OneCycleLR", "pct_start": 0.1, "loss": "CTC"}
    sources = {"src.train.main": inspect.getsource(train.main), "src.dataset.to_tensor": inspect.getsource(to_tensor),
               "src.charlm": inspect.getsource(charlm)}
    where, text = em.CODE_FACTS[name]
    assert text in sources[where]
    # Kodenya berubah (pengoptimal diganti, rugi lain, normalisasi lain, ...): ekspor harus berhenti, bukan menulis
    # kalimat lama. Berlaku untuk TIAP potongan yang dijaga, bukan hanya yang pertama.
    monkeypatch.setitem(em.CODE_FACTS, name, (where, "potongan yang sudah tidak ada"))
    with pytest.raises(SystemExit, match=f"kode tidak lagi cocok dengan kartu metode \\({name}\\)"):
        em.code_facts()


def tiny_checkpoint(path: Path, classes: int = 6) -> CRNN:
    model = CRNN(classes, channels=(2, 2, 2, 2), hidden=4)
    saved = torch.load(path, weights_only=False) if path.exists() else {"step": 1, "args": {}}
    saved["args"].update({"lr": 3e-4, "weight_decay": 1e-4, "clip": 5.0, "max_batch_columns": 64000, "packed": True})
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**saved, "model": model.state_dict(),
                "model_config": {"n_classes": classes, "channels": [2, 2, 2, 2], "hidden": 4}}, path)
    return model


def test_model_card_is_counted_from_the_built_model(tmp_path, monkeypatch):
    path = tmp_path / "run" / "last_snapshot.pt"
    model = tiny_checkpoint(path)
    card = em.model_card(path)
    total = sum(p.numel() for p in model.parameters())
    assert card["parameters"]["total"] == total == sum(v for k, v in card["parameters"].items() if k != "total")
    assert (card["classes"], card["channels"], card["hidden"]) == (6, [2, 2, 2, 2], 4)
    assert (card["conv_layers"], card["kernel"], card["lstm_layers"], card["bidirectional"]) == (7, 3, 2, True)
    assert (card["height"], card["feature_height"], card["width_stride"]) == (96, 3, 4)

    # Dua pengaman: keluaran model harus benar-benar selebar W // langkah, dan konstanta model harus sama dengan
    # konstanta data (tinggi 96, langkah lebar 4).
    with monkeypatch.context() as patched:
        patched.setattr(em.CRNN, "WIDTH_STRIDE", 8)
        with pytest.raises(SystemExit, match="tidak sesuai langkah lebar 8"):
            em.model_card(path)
    with monkeypatch.context() as patched:
        patched.setattr(em, "H", 64)
        with pytest.raises(SystemExit, match="tinggi atau langkah lebar model berbeda"):
            em.model_card(path)

    # model_config yang tidak cocok dengan bobotnya ditolak (bobot dimuat ketat).
    saved = torch.load(path, weights_only=False)
    saved["model_config"]["hidden"] = 8
    torch.save(saved, path)
    with pytest.raises(RuntimeError):
        em.model_card(path)


def compare_file(root: Path, a: str, b: str, cer_a: float, cer_b: float, ci: list[float], **extra) -> None:
    """Berkas pembanding seperti keluaran scripts/compare_runs.py: dua selang (baris dan halaman) yang berbeda, dan
    pengaturan bootstrap yang bukan bawaan skrip (2.000), supaya terlihat dari mana kartu mengutipnya."""
    folder = root / "out/compare"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{a}_vs_{b}.json").write_text(json.dumps({
        "a": {"key": a}, "b": {"key": b}, "lines": 3,
        "bootstrap": {"resamples": 2000, "seed": 0, "interval": "persentil 2,5-97,5"},
        "cer": {"a": cer_a, "b": cer_b, "diff": cer_a - cer_b, "bootstrap_clusters": {"ci95": ci},
                "bootstrap_lines": {"ci95": [edge / 2 for edge in ci]}}, **extra}), encoding="utf-8")


def manifest_of(root: Path) -> dict:
    return json.loads((root / "out/results/manifest.json").read_text(encoding="utf-8"))


def write_manifest(root: Path, manifest: dict) -> None:
    (root / "out/results/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def method_repo(root: Path) -> Path:
    """Repo palsu kartu data + yang dibutuhkan kartu metode: bobot model kecil, run kontrol, manifest hasil, berkas
    pembanding."""
    fake_repo(root, run=RUN)
    model = None
    for run in ("runA", RUN):
        model = tiny_checkpoint(root / "out/checkpoints" / run / "last_snapshot.pt")
    params = sum(p.numel() for p in model.parameters())
    write_log(root / "out/checkpoints" / RUN / "log.jsonl", [
        {"event": "start", "step": 0, "args": {}, "params": params, "device": "xpu"},
        {"step": 10, "val_cer": 0.05, "val_lines": 10}, {"step": 10, "event": "end"}])
    # Run kontrol perlakuan jarak: sama dengan run resmi, tanpa --track-prob.
    save_ckpt(root / "out/checkpoints/fase7_ctrl/last_snapshot.pt", 10, init="out/checkpoints/runA/last_snapshot.pt",
              extra_fonts="fonts/extra", augment="fase5", drop_space_prob=0.5)

    tokenizer = json.loads((root / "data/tokenizer.json").read_text(encoding="utf-8"))
    tokenizer["stats"] = {"lines": 2200, "fonts": ["a.ttf", "b.ttf", "c.ttf"], "prebase": ["U+A9BA", "U+A9BB"],
                          "before_base_rate": {"U+A9BB dirga mure": 1.0, "U+A9BA taling": 1.0, "U+A9BF cakra": 0.03},
                          "reorder_patterns": 370, "non_identity_patterns": 82, "harfbuzz_agreement": 0.99937}
    (root / "data/tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")

    def pipeline(key: str, config: str, status: str = "done") -> dict:
        return {"key": key, "label": key.upper(), "kind": "crnn", "status": status, "config": config}

    def metric(scope: str, key: str, cer: float, **extra) -> dict:
        return {"scope": scope, "pipeline": key, "lines": 50 if scope == "blind_50" else 3, "cer": cer, **extra}

    (root / "out/results").mkdir(parents=True)
    write_manifest(root, {
        "schema": 1, "generated": "2026-10-04T15:46:51", "limited": False, "official": "crnn_fase7_track",
        "pipelines": [pipeline("crnn_core", "crnn:fase5_core@1298 · 2 font · greedy"),
                      pipeline("crnn_fonts", "crnn:fase5_fonts@1298 · 3 font · greedy"),
                      pipeline("crnn_fonts_beam", "crnn:fase5_fonts@1298 · beam 16"),
                      pipeline("crnn_fase6_ctrl", "crnn:fase6_ctrl@1500 · 3 font · greedy"),
                      pipeline("crnn_fase7_track", "crnn:fase7_track@10 (lanjutan runA@6) · 3 font · greedy"),
                      pipeline("crnn_fase7_ctrl", "crnn:fase7_ctrl@10 · 3 font · greedy"),
                      pipeline("vlm_zeroshot", "vlm:frontier · zero-shot"),
                      pipeline("vlm_finetune", "konfigurasi vlm_finetune", "planned"),
                      pipeline("gpt", "konfigurasi gpt", "planned")],
        "gates": [{"code": "G1", "value": 0.0027, "target": 0.02, "passed": True},
                  {"code": "G2", "value": 0.012, "target": 0.05, "passed": True},
                  {"code": "G3", "value": 0.20, "target": 0.08, "passed": False},
                  {"code": "G4", "value": 1.0, "target": 1.0, "passed": True, "basis": "2.200 / 2.200 baris korpus (Fase 1)",
                   "source": "tests/test_tokenizer.py"}],
        "metrics": [metric("nusaaksara_745", "crnn_core", 0.60), metric("nusaaksara_745", "crnn_fonts", 0.40),
                    metric("nusaaksara_745", "crnn_fonts_beam", 0.35, better=5, worse=2),
                    metric("nusaaksara_745", "crnn_fase7_track", 0.20), metric("nusaaksara_745", "crnn_fase7_ctrl", 0.30),
                    metric("blind_50", "vlm_zeroshot", 0.90), metric("blind_50", "crnn_fonts", 0.40)],
        # Dua operasi preset (satu naik, satu turun) lalu dua operasi tiruan pindaian: urutan muncul bukan urutan besar.
        "ablation": [{"k": 0, "added": "(kontrol)", "G3": 0.75}, {"k": 1, "added": "rotate", "G3": 0.80},
                     {"k": 2, "added": "blur", "G3": 0.70}, {"k": 3, "added": "stroke", "G3": 0.65},
                     {"k": 4, "added": "tight", "G3": 0.50}],
    })
    # Perlakuan - kontrol, dan arah sebaliknya (kontrol dulu) untuk pasangan font.
    compare_file(root, "crnn_fase7_track", "crnn_fase7_ctrl", 0.20, 0.30, [-0.12, -0.07])
    compare_file(root, "crnn_core", "crnn_fonts", 0.60, 0.40, [0.15, 0.25])
    return root


def by_key(card: dict) -> dict:
    return {m["key"]: m for group in card["groups"] for m in group["methods"]}


def build(root: Path) -> dict:
    """Kartu repo palsu dengan font uji palsunya (salinan font inti pertama), bukan javatext mesin ini."""
    return em.build(root, RUN, root / "ujifont")


def test_build_lists_methods_with_computed_settings_and_evidence(tmp_path):
    root = method_repo(tmp_path)
    card = build(root)
    dump = json.dumps(card, ensure_ascii=False)
    assert SECRET not in dump  # teks label data nyata tidak pernah ikut
    assert card["schema"] == em.SCHEMA and card["results_generated"] == "2026-10-04T15:46:51"
    assert card["chain_complete"] is True
    assert card["official"] == {"run": RUN, "pipeline": "crnn_fase7_track",
                                "checkpoint": f"out/checkpoints/{RUN}/last_snapshot.pt", "step": 10}
    assert [g["key"] for g in card["groups"]] == ["ocr", "data", "training", "decoding", "evaluation"]
    assert card["planned"] == [{"key": "vlm_finetune", "label": "VLM_FINETUNE", "config": "konfigurasi vlm_finetune"},
                               {"key": "gpt", "label": "GPT", "config": "konfigurasi gpt"}]
    methods = by_key(card)
    assert len(methods) == sum(len(g["methods"]) for g in card["groups"])  # kunci unik
    for m in methods.values():
        assert m["kind"] in em.KINDS and m["status"] in em.STATUSES and m["summary"].strip() and m["name"].strip()
        assert all(len(pair) == 2 and all(isinstance(x, str) and x for x in pair) for pair in m["settings"])
        assert len({pair[0] for pair in m["settings"]}) == len(m["settings"])  # label pengaturan unik per butir
        assert (m["evidence"] is None) == (m["evidence_source"] is None)
        assert m["note"] is None or m["note"].strip()

    params = card["model"]["parameters"]
    assert params["total"] == sum(v for k, v in params.items() if k != "total")
    # Pengaturan tiap butir, lengkap: dari model kecil di checkpoint (kanal 2, tersembunyi 4, 6 kelas), konstanta kode,
    # dan argumen training repo palsu. Satu angka yang pindah butir atau tertukar harus menggagalkan test ini.
    assert methods["cnn"]["settings"] == [
        ["Lapis konvolusi", "7 (3×3, BatchNorm, ReLU)"], ["Kanal", "2, 2, 2, 2"],
        ["Parameter", f"{em.num(params['cnn'] + params['proj'])} (termasuk proyeksi ke 4 dimensi)"]]
    assert f"Tinggi citra {H} piksel dipampatkan menjadi 3, sedangkan lebarnya hanya diperkecil 4 kali" in methods["cnn"]["summary"]
    assert methods["bilstm"]["name"] == "BiLSTM (LSTM dua arah)"
    assert methods["bilstm"]["settings"] == [["Lapis", "2"], ["Unit per arah", "4"], ["Parameter", em.num(params["rnn"])]]
    assert methods["head"]["settings"] == [["Kelas", "6"], ["Parameter", em.num(params["head"])]]
    assert "4 karakter aksara Jawa" in methods["head"]["summary"] and "spasi, dan satu kelas kosong" in methods["head"]["summary"]
    assert len({params["cnn"], params["proj"], params["rnn"], params["head"]}) == 4  # supaya angka yang tertukar terlihat
    assert methods["ctc"]["settings"] == [["Kelas kosong", f"indeks {BLANK}"],
                                          ["Kolom per karakter, paling sedikit", em.dec(MIN_FRAMES_PER_TARGET)]]
    assert MIN_FRAMES_PER_TARGET != 1 and RENDER_SIZE_RANGE[0] < RENDER_SIZE_RANGE[1]
    assert methods["render"]["settings"] == [
        ["Ukuran huruf", f"{RENDER_SIZE_RANGE[0]} sampai {RENDER_SIZE_RANGE[1]} piksel, diundi per baris"],
        ["Tinggi citra akhir", f"{H} piksel"]]
    assert methods["render"]["note"] is None  # font repo palsu tidak punya cacat yang diketahui
    assert methods["greedy"]["settings"] == [] and methods["contrast"]["settings"] == []
    assert methods["visual_order"]["settings"] == [
        ["Tanda yang dipindah ke depan", "taling (U+A9BA), dirga mure (U+A9BB)"],
        ["Pola suku kata", "370 (82 berubah urutan)"], ["Sama dengan urutan HarfBuzz", "99,94% kemunculan suku kata"]]
    assert "pada 3 font" in methods["visual_order"]["summary"]
    # Nilai, dasar, dan sumber gerbang G4 dikutip dari manifest (ekspor hasil yang menetapkannya).
    assert methods["visual_order"]["evidence"] == ("Gerbang G4, 100%: bolak-balik urutan visual dan urutan Unicode diuji "
                                                   "pada 2.200 / 2.200 baris korpus (Fase 1).")
    assert methods["visual_order"]["evidence_source"] == "tests/test_tokenizer.py"

    # Data: label sintetis tidak disebut "pasti benar"; bukti font menyebut run mana yang dibandingkan.
    data = card["groups"][1]
    assert data["intro"].endswith("labelnya teks yang dirender itu sendiri, bukan anotasi manusia.") and "pasti" not in data["intro"]
    assert methods["fonts"]["summary"].startswith("Tiap baris dirender dengan satu font yang diundi dari 3 font. ")
    assert methods["fonts"]["settings"] == [["Font latih", "3 (2 inti, 1 tambahan)"]]
    assert methods["fonts"]["evidence"] == ("Pada run fase5_core lawan fase5_fonts (2 lawan 3 font, bukan model resmi): G3 "
                                            "60,00% → 40,00% (−20,00 poin; selang kepercayaan 95% per halaman −25,00 "
                                            "sampai −15,00).")
    assert methods["fonts"]["evidence_source"] == "out/compare/crnn_core_vs_crnn_fonts.json"
    assert methods["augment"]["settings"] == [["Preset", "fase5"], ["Operasi", f"{len(PHYSICAL_ORDER)}: {', '.join(PHYSICAL_ORDER)}"],
                                              ["Peluang tiap operasi", "0,5"]]
    assert set(em.AUGMENT_WORDS) == set(PHYSICAL_ORDER)  # operasi baru di src.augment wajib diberi kata di kartu
    assert methods["augment"]["summary"] == (
        "Citra bersih hasil render dirusak secara acak supaya mirip pindaian dan foto: diberi tepi kosong, dipotong rapat "
        "sampai tinta menyentuh tepi, dimiringkan sedikit, dilengkungkan halus, ditebal-tipiskan goresannya, diubah "
        "kontrasnya, diberi bayangan dan butiran kertas, diburamkan, diberi bising, dijadikan hitam-putih, diberi bintik, "
        "dan dikompresi. Tiap operasi diundi sendiri-sendiri, jadi tiap citra mendapat campuran yang berbeda.")
    # Ablasi: jumlah per kelompok operasi, bukan "penurunan terbesar"; langkah yang naik disebut di catatannya.
    assert methods["augment"]["evidence"] == (
        "Ablasi 5 run: G3 75,00% → 50,00%. 2 operasi preset bersama mengubahnya −5,00 poin; 2 operasi tiruan pindaian "
        "sesudahnya −20,00 poin (stroke −5,00 poin, tight −15,00 poin).")
    assert methods["augment"]["note"] == (
        "Operasi ditambahkan berurutan dan tiap run dilatih satu kali, jadi selisih satu langkah mencampur pengaruh "
        "operasinya dengan derau antar-run (menambahkan rotate malah menaikkan G3 5,00 poin). G3 adalah data uji: angka "
        "ini dibaca sebagai arah, bukan dasar memilih operasi.")
    assert methods["tracking"]["settings"] == [["Peluang", "0,5"], ["Jarak tambahan", "0 sampai 0,3 em"]]
    assert methods["tracking"]["evidence"] == ("Terhadap run kontrol fase7_ctrl (jumlah langkah sama, tanpa jarak): G3 "
                                               "30,00% → 20,00% (−10,00 poin; selang kepercayaan 95% per halaman −12,00 "
                                               "sampai −7,00).")
    assert methods["tracking"]["evidence_source"] == "out/compare/crnn_fase7_track_vs_crnn_fase7_ctrl.json"
    assert methods["drop_space"]["settings"] == [["Peluang", "0,5"]]  # tidak ada font berspasi sempit di repo palsu
    assert methods["drop_space"]["evidence"] == "2 dari 3 baris cetak nyata (66,7%) ditulis tanpa spasi."
    assert "rare" not in methods  # tidak dipakai run resmi dan tidak ada berkas pembandingnya

    assert methods["optimizer"]["settings"] == [["Laju belajar puncak", "0,0003"], ["Peluruhan bobot", "0,0001"],
                                                ["Ukuran batch", "8"], ["Langkah run resmi", "10"]]
    assert methods["onecycle"]["settings"] == [["Bagian naik", "10%"], ["Puncak", "0,0003"],
                                               ["Langkah direncanakan run resmi", "10"]]
    assert methods["onecycle"]["summary"] == ("Laju belajar naik selama 10% pertama dari langkah yang direncanakan sebuah "
                                              "run, lalu turun sampai langkah terakhirnya. Run resmi menyelesaikan "
                                              "siklusnya (10 langkah).")
    assert methods["clip"]["settings"] == [["Batas norma", "5"]]
    assert methods["batching"]["settings"] == [["Pengemasan LSTM", "ya"], ["Batas kolom per forward", "64.000"]]
    assert methods["staged"]["settings"] == [
        ["Rantai run", f"runA → {RUN}"], ["Jumlah langkah", "16"],
        ["Yang berubah", f"{RUN}: augmentasi fase5, 3 font (dari 2), buang spasi, jarak antar suku kata"],
        ["Perangkat", "grafis bawaan Intel (PyTorch XPU)"]]
    assert card["groups"][2]["intro"].startswith("Model dilatih bertahap di laptop tanpa kartu grafis NVIDIA. ")
    assert methods["staged"]["summary"].startswith("Model resmi bukan hasil satu kali latih. 2 run berurutan")

    beam = methods["beam_lm"]
    assert beam["status"] == "available" and ALPHA != BETA
    # Tanpa baris "Model bahasa": berkas LM tidak ada di repo palsu.
    assert beam["settings"] == [["Lebar beam", str(WIDTH)], ["Bobot model bahasa", em.dec(ALPHA)], ["Bonus panjang", em.dec(BETA)]]
    assert beam["evidence"] == ("Pada checkpoint fase5_fonts: G3 40,00% → 35,00% (5 baris membaik, 2 memburuk). "
                                "Belum diuji pada checkpoint resmi.")
    assert methods["gates"]["evidence"] == "G1 0,27% · G2 1,20% · G3 20,00% (belum tercapai) · G4 100%."
    assert TARGETS["G1"] != TARGETS["G2"]
    assert methods["gates"]["settings"] == [
        ["G1, render bersih", f"CER < {em.pct(TARGETS['G1'], 0)} pada {em.num(OFFICIAL_LINES)} baris uji, font javatext"],
        ["G2, render rusak berat", f"CER < {em.pct(TARGETS['G2'], 0)} pada {em.num(OFFICIAL_LINES)} baris uji, font javatext"],
        ["G3, cetakan nyata", f"CER < {em.pct(TARGETS['G3'], 0)} pada 3 baris"], ["G4, tokenizer", "bolak-balik 100%"]]
    # Batasan G1/G2 (CLAUDE.md: wajib disebut): font uji palsu = salinan font inti pertama, jadi sekeluarga.
    assert methods["gates"]["note"] == FAMILY_NOTE.format("G1 dan G2")
    # Pengaturan bootstrap dikutip dari berkas pembanding (2.000), bukan bawaan scripts/compare_runs.py.
    assert inspect.signature(compare_runs.compare).parameters["resamples"].default != 2000
    assert methods["bootstrap"]["settings"] == [["Pengambilan ulang", "2.000"], ["Selang", "persentil 2,5 sampai 97,5"],
                                                ["Satuan", "baris, dan halaman (2 halaman)"]]
    assert "secara acak 2.000 kali" in methods["bootstrap"]["summary"]
    assert "bukan keragaman antar-run" in methods["bootstrap"]["summary"]
    assert methods["cer"]["settings"] == [["Jarak", "Levenshtein per karakter"]]
    # Run kontrol per perlakuan: hanya pasangan yang kedua pipeline-nya ada di hasil.
    assert methods["control"]["settings"] == [["Kontrol font tambahan", "fase5_core (untuk fase5_fonts)"],
                                              ["Kontrol jarak antar suku kata", "fase7_ctrl (untuk fase7_track)"]]
    assert "(font tambahan dan jarak antar suku kata)" in methods["control"]["summary"]
    assert methods["ablation"]["settings"] == [["Run", "5"]] and "probe" not in methods
    assert methods["vlm_blind"]["status"] == "comparator" and methods["vlm_blind"]["settings"] == [["Baris", "50"]]
    # "official" = bagian dari model resmi; cara mengukur berstatus "used".
    assert {m["status"] for m in card["groups"][-1]["methods"]} == {"used", "comparator"}
    assert all(m["status"] == "official" for g in card["groups"][:3] for m in g["methods"])
    assert methods["vlm_blind"]["evidence"] == "CER 90,00% melawan 40,00% untuk CRNN saat itu (fase5_fonts)."


def test_optional_evidence_appears_only_when_its_files_exist(tmp_path):
    root = method_repo(tmp_path)
    manifest = manifest_of(root)
    manifest["metrics"].append({"scope": "nusaaksara_745", "pipeline": "crnn_fase7_track_rare", "lines": 3, "cer": 0.21})
    manifest["pipelines"].append({"key": "crnn_fase7_track_rare", "label": "R", "kind": "crnn", "status": "done", "config": "x"})
    write_manifest(root, manifest)
    rare_groups = {"groups": {"langka": {"recall": {"a": 0.5, "b": 0.0}, "precision": {"a": 0.4}}}}
    compare_file(root, "crnn_fase7_track_rare", "crnn_fase7_track", 0.21, 0.20, [-0.01, 0.02], rare=rare_groups)
    (root / "out/compare/spacing_synthetic.json").write_text(json.dumps(
        {"settings": {"tracking": [0.0, 0.1, 0.45], "lines": 301}}), encoding="utf-8")
    for k in range(5):
        (root / f"out/checkpoints/ablation_{k}").mkdir(parents=True)
        write_log(root / f"out/checkpoints/ablation_{k}/log.jsonl", [{"event": "start", "step": 0, "args": {"steps": 600, "seed": 0}}])
    lm = root / LM_PATH.relative_to(em.ROOT)
    lm.parent.mkdir(parents=True)
    lm.write_bytes(b"")
    # Berkas pembanding yang isinya untuk pasangan lain (nama berkasnya saja yang cocok) tidak dipakai.
    wrong = root / "out/compare/crnn_fase7_track_vs_crnn_fase7_ctrl.json"
    wrong.write_text(wrong.read_text(encoding="utf-8").replace('"crnn_fase7_ctrl"', '"crnn_lain"'), encoding="utf-8")
    methods = by_key(build(root))

    rare = methods["rare"]
    assert rare["status"] == "tested" and rare["settings"] == []
    assert rare["evidence"] == ("Pada 3 baris nyata, aksara langka yang terbaca benar naik dari 0,0% ke 50,0%, tetapi hanya "
                                "40,0% dari keluarannya benar, dan G3 tidak berbeda nyata (21,00% melawan 20,00%). Karena "
                                "itu tidak dipakai di run resmi.")
    assert rare["evidence_source"] == "out/compare/crnn_fase7_track_rare_vs_crnn_fase7_track.json"
    assert methods["control"]["settings"][-1] == ["Kontrol sisipan aksara langka", "fase7_track (untuk fase7_track_rare)"]
    # Ablasi yang log-nya ada: langkah per run, dan semua run berseed sama disebut di catatan.
    assert methods["ablation"]["settings"] == [["Run", "5"], ["Langkah per run", "600"]]
    assert methods["augment"]["note"].startswith("Operasi ditambahkan berurutan dan tiap run dilatih satu kali dengan seed yang sama, ")
    # Berkas model bahasa ada: order dan jumlah barisnya dibaca dari nama berkasnya.
    assert methods["beam_lm"]["settings"][-1] == ["Model bahasa", "n-gram karakter order 5, penghalusan Witten-Bell, 300.000 baris latih"]
    assert len(methods["beam_lm"]["settings"]) == 4
    # Tanpa berkas pembanding yang sah, selisihnya tetap dari manifest, hanya tanpa selang kepercayaan.
    assert methods["tracking"]["evidence"] == ("Terhadap run kontrol fase7_ctrl (jumlah langkah sama, tanpa jarak): G3 "
                                               "30,00% → 20,00% (−10,00 poin).")
    assert methods["tracking"]["evidence_source"] == "out/results/manifest.json"

    # Selang kepercayaan yang tidak memuat nol: selisih G3 disebut apa adanya, bukan "tidak berbeda nyata".
    manifest["metrics"][-1]["cer"] = 0.26
    write_manifest(root, manifest)
    compare_file(root, "crnn_fase7_track_rare", "crnn_fase7_track", 0.26, 0.20, [0.03, 0.09], rare=rare_groups)
    assert "dan G3 berubah +6,00 poin (26,00% melawan 20,00%)" in by_key(build(root))["rare"]["evidence"]
    # F10: berkas pembanding dari ekspor hasil yang lain (CER-nya bukan CER manifest sekarang) tidak dikutip.
    manifest["metrics"][-1]["cer"] = 0.27
    write_manifest(root, manifest)
    assert "rare" not in by_key(build(root))

    assert methods["probe"]["settings"] == [["Jarak yang diuji", "0; 0,1; 0,45 em"], ["Baris", "301"]]
    assert methods["probe"]["evidence"] is None  # laporan tanpa hasil per run
    assert methods["probe"]["note"] is None      # dan tanpa nama font perendernya

    # Baris yang gagal dirender dibuang evaluasinya: yang disebut adalah jumlah yang benar-benar dibaca. Buktinya
    # spasi palsu pada jarak terbesar yang masih di rentang latih run resmi (--track-max 0,3) lawan run kontrolnya,
    # lalu pada jarak terbesar di luar rentang itu.
    def rows(rates):
        return [{"tracking": tracking, "false_per_boundary": rate} for tracking, rate in rates]

    spacing = {"settings": {"tracking": [0.0, 0.15, 0.3, 0.45, 0.6], "lines": 301}, "fonts": {"javatext.ttf": {"lines": 300, "results": {
        RUN: {"tanpa-spasi": rows([(0.0, 0.0), (0.15, 0.001), (0.3, 0.012), (0.6, 0.675), (0.45, 0.04)]), "berspasi": rows([(0.3, 0.5)])},
        "fase7_ctrl": {"tanpa-spasi": rows([(0.0, 0.0), (0.15, 0.6), (0.3, 0.925), (0.6, 0.876)])}}}}}
    (root / "out/compare/spacing_synthetic.json").write_text(json.dumps(spacing), encoding="utf-8")
    probe = by_key(build(root))["probe"]
    assert probe["settings"] == [["Jarak yang diuji", "0; 0,15; 0,3; 0,45; 0,6 em"], ["Baris", "300 (dari 301 yang diambil)"]]
    inside = ("Baris tanpa spasi yang direnggangkan 0,3 em: 1,2 spasi palsu per 100 batas suku kata pada model resmi, 92,5 "
              "pada run kontrol fase7_ctrl.")
    assert probe["evidence"] == inside + " Di luar rentang jarak yang dilatihkan (0,6 em) model resmi kembali mengeluarkan 67,5."
    assert probe["evidence_source"] == "out/compare/spacing_synthetic.json"
    assert probe["note"] == ("Citranya dirender dengan font uji javatext, yang sekeluarga dengan font latih "
                             "NotoSansJavanese-Regular dan Salinan, jadi hasilnya belum tentu berlaku untuk bentuk huruf lain.")
    assert em.spacing_note(spacing, RUN, "fase7_ctrl", 0.15) == (
        "Baris tanpa spasi yang direnggangkan 0,15 em: 0,1 spasi palsu per 100 batas suku kata pada model resmi, 60 pada "
        "run kontrol fase7_ctrl. Di luar rentang jarak yang dilatihkan (0,6 em) model resmi kembali mengeluarkan 67,5.")
    assert em.spacing_note(spacing, RUN, "fase7_ctrl", 0.6) == (
        "Baris tanpa spasi yang direnggangkan 0,6 em: 67,5 spasi palsu per 100 batas suku kata pada model resmi, 87,6 pada "
        "run kontrol fase7_ctrl.")
    # Tanpa run kontrol, tanpa jarak saat latih, run yang tidak ikut dievaluasi, atau laporan tanpa hasil: tidak ada bukti.
    assert em.spacing_note(spacing, RUN, None, 0.3) is None and em.spacing_note(spacing, RUN, "fase7_ctrl", 0.0) is None
    assert em.spacing_note(spacing, "run_lain", "fase7_ctrl", 0.3) is None and em.spacing_note(spacing, RUN, "run_lain", 0.3) is None
    assert em.spacing_note({"settings": {}}, RUN, "fase7_ctrl", 0.3) is None
    # Kedua run harus dibaca pada jarak yang sama: kontrol yang hanya dievaluasi sampai 0,15 em tidak dibandingkan
    # dengan model resmi pada 0,3 em.
    partial = json.loads(json.dumps(spacing))
    partial["fonts"]["javatext.ttf"]["results"]["fase7_ctrl"]["tanpa-spasi"] = rows([(0.0, 0.0), (0.15, 0.6)])
    assert em.spacing_note(partial, RUN, "fase7_ctrl", 0.3) is None
    # Nilai kecil yang bukan nol tidak dibulatkan menjadi "0" (terukur 2026-10-04: 1 spasi palsu di 5.209 batas).
    partial["fonts"]["javatext.ttf"]["results"][RUN]["tanpa-spasi"] = rows([(0.15, 0.00019)])
    assert em.spacing_note(partial, RUN, "fase7_ctrl", 0.15) == (
        "Baris tanpa spasi yang direnggangkan 0,15 em: 0,02 spasi palsu per 100 batas suku kata pada model resmi, 60 pada "
        "run kontrol fase7_ctrl.")
    # F06: run yang terdaftar sebagai kontrol tetapi checkpoint-nya ternyata dilatih DENGAN jarak bukan kontrol: kartu
    # tidak membandingkan dengannya, walau laporan evaluasinya memuat run itu.
    save_ckpt(root / "out/checkpoints/fase7_ctrl/last_snapshot.pt", 10, track_prob=0.5, track_max=0.3)
    unverified = by_key(build(root))
    assert unverified["probe"]["evidence"] is None and unverified["tracking"]["evidence"] is None
    assert unverified["probe"]["settings"] == probe["settings"]


def data_methods(root: Path, run: str, args: dict, fonts: dict | None = None) -> dict:
    """Butir kelompok data untuk sebuah resep, dengan fakta font bawaan: 2 font inti, tanpa cacat, tanpa spasi sempit."""
    facts = {"train": 2, "core": 2, "extra": 0, "measured": 2, "current": True, "defects": [], "narrow": 0, "recorded": {}}
    real = em.nusaaksara_card(root / "data/real/nusaaksara/labels.tsv")
    group = em.data_group(run, args, {**facts, **(fonts or {})}, real, em.Evidence(root), root)
    return {m["key"]: m for m in group["methods"]}


def test_data_sentences_follow_the_recipe_and_the_fonts(tmp_path):
    root = method_repo(tmp_path)

    two = data_methods(root, RUN, {"augment": "noise+rotate"})["augment"]
    assert two["summary"] == ("Citra bersih hasil render dirusak secara acak supaya mirip pindaian dan foto: dimiringkan "
                              "sedikit dan diberi bising. Tiap operasi diundi sendiri-sendiri, jadi tiap citra mendapat "
                              "campuran yang berbeda.")
    assert ["Operasi", "2: rotate, noise"] in two["settings"] and ["Peluang tiap operasi", "0,5"] in two["settings"]
    # Preset G2: delapan operasi, semuanya selalu dikenakan (p = 1), tanpa operasi pindaian seperti potong rapat.
    heavy = data_methods(root, RUN, {"augment": "heavy"})["augment"]
    assert heavy["summary"].endswith("diberi bising, dan dikompresi. Semua operasi dikenakan pada tiap citra.")
    assert "dipotong rapat" not in heavy["summary"] and ["Peluang tiap operasi", "1"] in heavy["settings"]

    # Run tanpa font tambahan: tidak ada klaim tentang variasi font, walau berkas pembandingnya ada.
    plain = data_methods(root, RUN, {"augment": "none"})
    assert set(plain) == {"render", "fonts"}  # tanpa augmentasi, jarak, buang spasi, dan sisipan
    assert plain["fonts"]["settings"] == [["Font latih", "2 (2 inti, 0 tambahan)"]]
    assert plain["fonts"]["evidence"] is None and plain["fonts"]["evidence_source"] is None
    recorded = {"fase5_core": 2, "fase5_fonts": 10}
    # ... juga bila ekspor hasil mencatat jumlah font kedua run uji itu: run ini sendiri tidak memakai font tambahan.
    assert data_methods(root, RUN, {"augment": "none"}, {"recorded": recorded})["fonts"]["evidence"] is None
    # Run perlakuan font itu sendiri yang resmi: buktinya miliknya, tanpa "bukan model resmi".
    own = data_methods(root, "fase5_fonts", {"augment": "none"}, {"train": 10, "extra": 8, "measured": 10, "recorded": recorded})
    assert own["fonts"]["evidence"].startswith("Pada run fase5_core lawan fase5_fonts (2 lawan 10 font): G3 60,00% → 40,00% ")
    # Tanpa catatan jumlah font kedua run di ekspor hasil, kalimat "N lawan M font" tidak dikarang dari folder.
    assert data_methods(root, RUN, {"augment": "none"}, {"train": 3, "extra": 1, "measured": 3})["fonts"]["evidence"] is None

    # F01: font latih bercacat disebut di butir render, dengan jumlahnya.
    flawed = data_methods(root, RUN, {"augment": "none"}, {"train": 10, "extra": 8, "measured": 10, "defects": ["BasaJan", "Carakan"]})
    assert flawed["render"]["note"] == (
        "2 dari 10 font latih punya cacat yang diketahui (BasaJan dan Carakan): pada font itu sebagian citra tidak sepadan "
        "dengan labelnya, atau pasangannya tidak menumpuk di lingkungan training. Rinciannya di halaman Dataset, bagian "
        "Font per peran.")
    # F08: folder font sudah berubah sejak run dilatih: jumlah yang tercatat tetap disebut, dan ukuran pada folder
    # sekarang diberi keterangan.
    moved = data_methods(root, RUN, {"augment": "none", "drop_space_prob": 0.5},
                         {"train": 10, "extra": 8, "measured": 9, "current": False, "defects": ["Carakan"], "narrow": 3})
    assert moved["fonts"]["summary"].startswith("Tiap baris dirender dengan satu font yang diundi dari 10 font. ")
    assert moved["fonts"]["settings"] == [["Font latih", "10 (tercatat saat run dilatih; folder font sekarang berisi 9)"]]
    assert moved["render"]["note"].startswith("1 dari 9 font latih yang ada di folder sekarang (run ini dilatih dengan 10 "
                                              "font) punya cacat yang diketahui (Carakan): ")
    # F14: spasi juga selalu dibuang pada font berspasi sempit, jadi porsinya lebih besar dari peluangnya:
    # 0,5 + 0,5 x 3/9 = 67%.
    assert moved["drop_space"]["settings"] == [
        ["Peluang", "0,5"], ["Selalu, pada font berspasi sempit", "3 dari 9 font yang ada di folder sekarang (run ini "
                                                                 "dilatih dengan 10 font)"],
        ["Porsi baris tanpa spasi", "sekitar 67%"]]
    assert "spasi selalu dibuang" in moved["drop_space"]["summary"]
    narrow = data_methods(root, RUN, {"augment": "none", "drop_space_prob": 0.5}, {"train": 10, "extra": 8, "measured": 10, "narrow": 3})
    assert narrow["drop_space"]["settings"] == [["Peluang", "0,5"], ["Selalu, pada font berspasi sempit", "3 dari 10 font"],
                                                ["Porsi baris tanpa spasi", "sekitar 65%"]]

    # Run resmi yang memakai sisipan aksara langka: butirnya bagian dari model resmi dengan peluangnya, dan kalimat
    # "diuji lalu tidak dipakai" tidak boleh muncul walau berkas pembanding uji itu ada.
    manifest = manifest_of(root)
    manifest["metrics"].append({"scope": "nusaaksara_745", "pipeline": "crnn_fase7_track_rare", "lines": 3, "cer": 0.21})
    write_manifest(root, manifest)
    compare_file(root, "crnn_fase7_track_rare", "crnn_fase7_track", 0.21, 0.20, [-0.01, 0.02], rare={"groups": {"langka": {
        "recall": {"a": 0.5, "b": 0.0}, "precision": {"a": 0.4}}}})
    rare = data_methods(root, "fase7_track_rare", {"augment": "none", "rare_insert_prob": 0.3, "rare_opener_prob": 0.15})["rare"]
    assert (rare["status"], rare["evidence"], rare["evidence_source"]) == ("official", None, None)
    assert rare["settings"] == [["Peluang sisip", "0,3"], ["Peluang pembuka", "0,15"]]
    # Hanya pembuka baris: peluang sisip tetap ditulis (nol), bukan galat.
    opener = data_methods(root, "fase7_track_rare", {"augment": "none", "rare_opener_prob": 0.15})["rare"]
    assert opener["settings"] == [["Peluang sisip", "0"], ["Peluang pembuka", "0,15"]]
    # Run lain yang BUKAN kontrol uji sisipan itu tidak mengutip hasilnya.
    assert "rare" not in data_methods(root, "fase7_ctrl", {"augment": "none"})
    assert data_methods(root, RUN, {"augment": "none"})["rare"]["status"] == "tested"
    # Run kontrol uji itu sendiri, bila kelak dilatih DENGAN sisipan: butirnya "dipakai", jadi hasil uji yang berakhir
    # "karena itu tidak dipakai di run resmi" tidak dikutip.
    used = data_methods(root, RUN, {"augment": "none", "rare_insert_prob": 0.3})["rare"]
    assert (used["status"], used["evidence"], used["evidence_source"]) == ("official", None, None)

    # F06: bukti jarak hanya terhadap run kontrol yang memang dilatih TANPA jarak. fase7_track_rare (jarak + sisipan)
    # tidak punya kontrol jarak sendiri: fase7_track adalah kontrol sisipannya dan dilatih dengan jarak.
    tracked = {"augment": "none", "track_prob": 0.5, "track_max": 0.3}
    assert data_methods(root, RUN, tracked)["tracking"]["evidence"].startswith("Terhadap run kontrol fase7_ctrl (jumlah langkah sama, tanpa jarak): ")
    both = data_methods(root, "fase7_track_rare", {**tracked, "rare_insert_prob": 0.3})["tracking"]
    assert both["evidence"] is None and both["settings"] == [["Peluang", "0,5"], ["Jarak tambahan", "0 sampai 0,3 em"]]
    assert em.tracking_control(root, "fase7_track_rare") is None and em.tracking_control(root, RUN) == "fase7_ctrl"
    # Run kontrol yang ternyata dilatih dengan jarak, atau yang checkpoint-nya tidak ada: tidak dipakai.
    save_ckpt(root / "out/checkpoints/fase7_ctrl/last_snapshot.pt", 10, track_prob=0.5, track_max=0.3)
    assert em.tracking_control(root, RUN) is None and data_methods(root, RUN, tracked)["tracking"]["evidence"] is None
    (root / "out/checkpoints/fase7_ctrl/last_snapshot.pt").unlink()
    assert em.tracking_control(root, RUN) is None


def test_ablation_evidence_does_not_rank_single_steps():
    def rows(*steps):
        return [{"added": name, "G3": value} for name, value in steps]

    # Operasi preset dan tiruan pindaian bercampur urutannya: tiap langkah disebut apa adanya, tanpa dijumlahkan.
    mixed = em.ablation_evidence(rows(("(kontrol)", 0.75), ("tight", 0.60), ("rotate", 0.58), ("stroke", 0.50)), {0, 1})
    assert mixed[0] == ("Ablasi 4 run: G3 75,00% → 50,00%. Selisih tiap langkah: tight −15,00 poin, rotate −2,00 poin, "
                        "stroke −8,00 poin.")
    # Tidak ada langkah yang naik: catatannya tidak menyebut kenaikan. Seed berbeda-beda: tidak disebut sama.
    assert mixed[1] == ("Operasi ditambahkan berurutan dan tiap run dilatih satu kali, jadi selisih satu langkah mencampur "
                        "pengaruh operasinya dengan derau antar-run. G3 adalah data uji: angka ini dibaca sebagai arah, "
                        "bukan dasar memilih operasi.")
    only_preset = em.ablation_evidence(rows(("(kontrol)", 0.75), ("rotate", 0.80), ("blur", 0.70)), {0})
    assert only_preset[0] == "Ablasi 3 run: G3 75,00% → 70,00%. Selisih tiap langkah: rotate +5,00 poin, blur −10,00 poin."
    assert "dengan seed yang sama" in only_preset[1] and "(menambahkan rotate malah menaikkan G3 5,00 poin)" in only_preset[1]
    # Satu baris, atau baris tanpa G3: tidak ada bukti.
    assert em.ablation_evidence(rows(("(kontrol)", 0.75)), {0}) is None
    assert em.ablation_evidence(rows(("(kontrol)", 0.75), ("rotate", None)), {0}) is None
    assert em.ablation_evidence([], set()) is None


def test_fonts_are_counted_from_the_log_then_the_results_then_the_folder(tmp_path, monkeypatch):
    root = method_repo(tmp_path)
    links, _ = em.chain(RUN, root)
    charset = [" ", KA, GA]
    recorded = em.recorded_fonts(root)
    assert recorded == {"fase5_core": 2, "fase5_fonts": 3, "fase6_ctrl": 3, "fase7_track": 3, "fase7_ctrl": 3}

    facts = em.font_facts(root, links[-1], charset, recorded)
    assert (facts["train"], facts["core"], facts["extra"], facts["source"], facts["measured"], facts["current"]) == (3, 2, 1, "results", 3, True)
    assert facts["defects"] == [] and facts["narrow"] == 0
    assert [path.name for path in facts["paths"]] == [font.name for font in TRAIN_FONTS] + ["Salinan.ttf"]
    # Run tanpa catatan di ekspor hasil: dari folder sekarang.
    first = em.font_facts(root, links[0], charset, recorded)
    assert (first["train"], first["source"], first["current"]) == (2, "folder", True)
    # Log run yang mencatat daftar fontnya didahulukan dari keduanya.
    names = [font.name for font in TRAIN_FONTS] + ["Salinan.ttf", "SudahDihapus.ttf"]
    write_log(links[-1]["path"].parent / "log.jsonl", [{**start(0), "fonts": names}, {"step": 10, "event": "end"}])
    logged = em.font_facts(root, links[-1], charset, recorded)
    assert (logged["train"], logged["extra"], logged["source"], logged["measured"], logged["current"]) == (4, 2, "log", 3, False)

    # Cacat = font di daftar cacat yang diketahui atau yang tidak punya glyph sebuah karakter charset; spasi sempit =
    # aturan src.dataset (celah kata hasil shaping di bawah ambang).
    monkeypatch.setattr(em, "FONT_DEFECTS", ("Salinan.ttf",))
    monkeypatch.setattr(em, "space_ratio", lambda path: 0.05 if Path(path).name == TRAIN_FONTS[1].name else 0.2)
    measured = em.font_facts(root, links[-1], charset + ["꧎"], recorded)  # U+A9CE: tidak ada di font mana pun
    assert measured["defects"] == sorted(path.stem for path in measured["paths"]) and measured["narrow"] == 1
    assert em.font_facts(root, links[-1], charset, recorded)["defects"] == ["Salinan"]

    # Font uji yang sekeluarga dengan font latih: salinan font inti pertama (73 dari 73 lebar maju sama).
    gates = [{"code": "G1", "fonts": ["javatext.ttf"]}, {"code": "G2", "fonts": ["javatext.ttf"]}, {"code": "G3", "fonts": []}]
    twins = em.family_twins(measured["paths"], gates, root / "ujifont")
    assert [(twin["test"], twin["train"], twin["same"], twin["of"]) for twin in twins] == [
        ("javatext", TRAIN_FONTS[0].stem, 73, 73), ("javatext", "Salinan", 73, 73)]
    assert em.family_note(twins, "G1 dan G2") == FAMILY_NOTE.format("G1 dan G2")
    # Font uji tidak ada di mesin ini, atau tidak sekeluarga dengan font latih mana pun: tidak ada catatan.
    assert em.family_twins(measured["paths"], gates, root / "tidak-ada") == [] and em.family_note([], "G1 dan G2") is None
    assert em.family_twins([TRAIN_FONTS[1]], gates, root / "ujifont") == []
    # Yang diperiksa hanya font uji gerbang SINTETIS: javatext yang ada di mesin tetapi tidak dipakai G1/G2 run ini
    # tidak ikut, dan gerbang data nyata tidak punya font uji.
    others = [{"code": "G1", "fonts": ["lain.ttf"]}, {"code": "G3", "fonts": ["javatext.ttf"]}]
    assert em.family_twins(measured["paths"], others, root / "ujifont") == []
    assert em.probe_note([]) is None


def test_gates_and_beam_evidence_follow_what_the_results_contain(tmp_path):
    root = method_repo(tmp_path)
    manifest = manifest_of(root)
    tokenizer = json.loads((root / "data/tokenizer.json").read_text(encoding="utf-8"))
    real = em.nusaaksara_card(root / "data/real/nusaaksara/labels.tsv")
    model = em.model_card(root / "out/checkpoints" / RUN / "last_snapshot.pt")

    def evaluation(twins=(), fonts=()):
        group = em.evaluation_group(RUN, {"track_max": 0.3}, real, em.Evidence(root), root, list(twins), list(fonts))
        return {m["key"]: m for m in group["methods"]}

    # Hasil tanpa gerbang G4 (ekspor lama): tokenizer tidak diberi bukti, dan G4 tidak disebut di daftar gerbang.
    write_manifest(root, {**manifest, "gates": manifest["gates"][:3]})
    order = {m["key"]: m for m in em.ocr_group(model, tokenizer, em.Evidence(root))["methods"]}["visual_order"]
    assert order["evidence"] is None and order["evidence_source"] is None
    gates = evaluation()["gates"]
    assert gates["evidence"] == "G1 0,27% · G2 1,20% · G3 20,00% (belum tercapai)."
    # Tanpa font uji yang diketahui dan tanpa kembaran: baris gerbang tidak menyebut font, dan tidak ada catatan.
    assert gates["settings"][0] == ["G1, render bersih", f"CER < {em.pct(TARGETS['G1'], 0)} pada {em.num(OFFICIAL_LINES)} baris uji"]
    assert gates["note"] is None
    # Gerbang G4 tanpa sumber di manifest: manifest itu sendiri sumbernya.
    bare = dict(manifest["gates"][3], source=None)
    write_manifest(root, {**manifest, "gates": manifest["gates"][:3] + [bare]})
    order = {m["key"]: m for m in em.ocr_group(model, tokenizer, em.Evidence(root))["methods"]}["visual_order"]
    assert order["evidence_source"] == "out/results/manifest.json"

    # Charset tanpa spasi, dan dengan dua karakter di luar blok Jawa: kalimat lapisan keluaran mengikutinya.
    def head(codes):
        return {m["key"]: m for m in em.ocr_group(model, {**tokenizer, "charset": codes}, em.Evidence(root))["methods"]}["head"]
    assert "2 karakter aksara Jawa (aksara, sandhangan, angka, pada), dan satu kelas kosong." in head(["U+A98F", "U+A992"])["summary"]
    assert "1 karakter aksara Jawa (aksara, sandhangan, angka, pada), 2 karakter lain, dan satu kelas kosong." in head(
        ["U+0020", "U+002D", "U+A98F"])["summary"]
    # Model satu arah (bukan arsitektur proyek ini): butirnya tidak menyebut "dua arah".
    one_way = {m["key"]: m for m in em.ocr_group({**model, "bidirectional": False}, tokenizer, em.Evidence(root))["methods"]}["bilstm"]
    assert one_way["name"] == "LSTM" and "kanan ke kiri" not in one_way["summary"]

    # Tanpa berkas pembanding sama sekali: jumlah pengambilan ulang = bawaan scripts/compare_runs.py, selang tidak disebut.
    write_manifest(root, manifest)
    for path in (root / "out/compare").glob("*_vs_*.json"):
        path.unlink()
    resamples = inspect.signature(compare_runs.compare).parameters["resamples"].default
    assert evaluation()["bootstrap"]["settings"] == [["Pengambilan ulang", em.num(resamples)], ["Satuan", "baris, dan halaman (2 halaman)"]]
    # Hasil tanpa satu pun pasangan perlakuan-kontrol: butir run kontrol tidak ada.
    write_manifest(root, {**manifest, "pipelines": [p for p in manifest["pipelines"] if p["key"] in ("crnn_fase7_track", "vlm_zeroshot")]})
    assert "control" not in evaluation()
    # Pipeline kontrol yang baru direncanakan (belum ada hasilnya) tidak disebut run kontrol.
    planned = [dict(p, status="planned") if p["key"] == "crnn_fase7_ctrl" else p for p in manifest["pipelines"]]
    write_manifest(root, {**manifest, "pipelines": planned})
    assert evaluation()["control"]["settings"] == [["Kontrol font tambahan", "fase5_core (untuk fase5_fonts)"]]

    # Beam + LM sudah diukur pada checkpoint resmi: angka itu yang dikutip, bukan angka fase5_fonts.
    extra = [{"scope": "nusaaksara_745", "pipeline": "crnn_fase7_track_beam", "lines": 3, "cer": 0.18, "better": 2, "worse": 1}]
    write_manifest(root, {**manifest, "metrics": manifest["metrics"] + extra})
    beam = em.decoding_group("crnn_fase7_track", em.Evidence(root), root)["methods"][0]
    assert beam["evidence"] == "Pada checkpoint resmi: G3 20,00% → 18,00% (2 baris membaik, 1 memburuk)."
    # Pipeline resmi = fase5_fonts sendiri: hasilnya memang milik checkpoint resmi.
    write_manifest(root, manifest)
    own = em.decoding_group("crnn_fonts", em.Evidence(root), root)["methods"][0]
    assert own["evidence"] == "Pada checkpoint resmi: G3 40,00% → 35,00% (5 baris membaik, 2 memburuk)."
    # Tidak ada hasil beam sama sekali: butirnya tetap ada, tanpa bukti.
    write_manifest(root, {**manifest, "metrics": [m for m in manifest["metrics"] if not m["pipeline"].endswith("_beam")]})
    none = em.decoding_group("crnn_fase7_track", em.Evidence(root), root)["methods"][0]
    assert (none["evidence"], none["evidence_source"], none["status"]) == (None, None, "available")


def test_build_stops_when_results_do_not_belong_to_the_run(tmp_path):
    root = method_repo(tmp_path)
    path = root / "out/results/manifest.json"
    manifest = manifest_of(root)

    write_manifest(root, {**manifest, "official": "crnn_fonts"})
    with pytest.raises(SystemExit, match="menetapkan crnn_fonts sebagai resmi, bukan crnn_fase7_track"):
        build(root)
    # Manifest lama tanpa kunci "official" berarti pipeline resminya crnn_fonts (aturan yang sama dengan web), jadi
    # kartu untuk run lain tetap ditolak.
    write_manifest(root, {key: value for key, value in manifest.items() if key != "official"})
    with pytest.raises(SystemExit, match="menetapkan crnn_fonts sebagai resmi, bukan crnn_fase7_track"):
        build(root)
    write_manifest(root, {**manifest, "limited": True})
    with pytest.raises(SystemExit, match="--limit"):
        build(root)
    path.unlink()
    with pytest.raises(SystemExit, match="export_results.py"):
        build(root)
    write_manifest(root, manifest)

    # Tokenizer yang jumlah karakternya tidak cocok dengan lapisan keluaran checkpoint: bukan pasangan yang sama.
    tokenizer_path = root / "data/tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer_path.write_text(json.dumps({**tokenizer, "charset": tokenizer["charset"][:-1]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="jumlah kelas checkpoint berbeda"):
        build(root)
    tokenizer_path.write_text(json.dumps(tokenizer), encoding="utf-8")

    # Laporan gerbang resmi run itu tidak ada: font uji tidak diketahui, kartu tidak ditulis.
    report = root / "out/eval" / f"{RUN}_G1_10k.json"
    saved = report.read_text(encoding="utf-8")
    report.unlink()
    with pytest.raises(SystemExit, match="laporan G1 .* tidak ada"):
        build(root)
    report.write_text(saved, encoding="utf-8")

    # Jumlah parameter di log training berbeda dari model yang dibangun: checkpoint dan log bukan run yang sama.
    write_log(root / "out/checkpoints" / RUN / "log.jsonl", [{"event": "start", "step": 0, "args": {}, "params": 7, "device": "cpu"}])
    with pytest.raises(SystemExit, match="mencatat 7 parameter"):
        build(root)


def test_training_sentences_follow_the_chain_and_the_logged_device(tmp_path):
    root = method_repo(tmp_path)
    links, complete = em.chain(RUN, root)
    code = em.code_facts()
    assert complete is True

    def training(chain_links, device="xpu", chain_complete=True, counts=None):
        group = em.training_group(chain_links, chain_complete, code, device, counts or [2] * len(chain_links))
        return group, {m["key"]: m for m in group["methods"]}

    # Satu run tanpa leluhur, dilatih di CPU: tidak boleh disebut "bertahap" atau "tanpa kartu grafis NVIDIA".
    lonely = [{**links[-1], "args": {**links[-1]["args"], "init": ""}}]
    alone, methods = training(lonely, "cpu")
    assert alone["intro"].startswith("Model dilatih dalam satu run di CPU, tanpa kartu grafis. ")
    assert methods["staged"]["summary"] == "Model resmi dilatih dalam satu run dari bobot acak."
    assert dict(methods["staged"]["settings"]) == {"Rantai run": RUN, "Jumlah langkah": "10", "Perangkat": "CPU"}
    # Perangkat lain atau tidak tercatat: kalimatnya tidak menyebut perangkat sama sekali.
    assert training(links, "cuda")[0]["intro"].startswith("Model dilatih bertahap. ")
    unknown, methods = training(links, None)
    assert unknown["intro"].startswith("Model dilatih bertahap. ")
    assert dict(methods["staged"]["settings"])["Perangkat"] == "tidak tercatat"

    # F07: checkpoint leluhur sudah dihapus. Rantai yang terbaca hanya ujungnya, tetapi model itu TIDAK dilatih dari
    # bobot acak: itu yang disebut, dan jumlah langkahnya batas bawah.
    cut, methods = training(links[-1:], "xpu", chain_complete=False)
    assert cut["intro"].startswith("Model dilatih bertahap di laptop tanpa kartu grafis NVIDIA. ")
    assert methods["staged"]["summary"].startswith("Model resmi bukan hasil satu kali latih: checkpoint-nya melanjutkan run "
                                                   "lain, tetapi checkpoint awal rantainya tidak ada di mesin ini")
    assert dict(methods["staged"]["settings"]) == {"Rantai run": f"… → {RUN}", "Jumlah langkah": "paling sedikit 10",
                                                   "Perangkat": "grafis bawaan Intel (PyTorch XPU)"}
    # Satu run yang dimulai dari bobot yang ada (args.init terisi) dan rantainya dianggap lengkap: bukan "bobot acak".
    seeded, methods = training(links[-1:], "xpu")
    assert methods["staged"]["summary"] == "Model resmi dilatih dalam satu run, dimulai dari bobot yang sudah ada."
    assert seeded["intro"].startswith("Model dilatih bertahap di laptop")

    # F09: siklus laju belajar dihitung untuk langkah yang DIRENCANAKAN. Run yang dihentikan lebih awal tidak
    # menyelesaikannya, dan itu disebut baik untuk run resmi maupun run sebelumnya di rantai.
    def planned(link, steps):
        return {**link, "args": {**link["args"], "steps": steps}}

    _, methods = training([planned(links[0], 60), links[-1]])
    assert methods["onecycle"]["summary"].endswith("Run resmi menyelesaikan siklusnya (10 langkah). 1 dari 1 run sebelumnya di "
                                                   "rantai (runA) dihentikan sebelum siklusnya selesai.")
    _, methods = training([links[0], planned(links[-1], 1500)])
    assert methods["onecycle"]["summary"].endswith("Run resmi dihentikan di langkah 10 dari 1.500 yang direncanakan, sebelum "
                                                   "siklusnya selesai.")
    assert ["Langkah direncanakan run resmi", "1.500"] in methods["onecycle"]["settings"]

    # Run eksperimen --no-pack: kartu menyebut LSTM tidak dikemas.
    loose = [{**link, "args": {**link["args"], "packed": False}} for link in links]
    assert training(loose)[1]["batching"]["settings"][0] == ["Pengemasan LSTM", "tidak"]

    # F07, dari ujung ke ujung: checkpoint awal rantai dihapus dari mesin. Kartu mencatat rantainya tidak lengkap, dan
    # butirnya tidak mengaku "dari bobot acak".
    assert build(root)["chain_complete"] is True
    (root / "out/checkpoints/runA/last_snapshot.pt").unlink()
    cut_card = build(root)
    assert cut_card["chain_complete"] is False
    staged = by_key(cut_card)["staged"]
    assert staged["summary"].startswith("Model resmi bukan hasil satu kali latih: ") and "bobot acak" not in staged["summary"]
    assert dict(staged["settings"])["Rantai run"] == f"… → {RUN}"


def test_recipe_changes_name_what_was_added_removed_or_left_alone():
    def link(run, **args):
        return {"run": run, "args": args}

    links = [link("a", augment="fase5", drop_space_prob=0.5, track_prob=0.5), link("b", augment="fase5", drop_space_prob=0.5),
             link("c", augment="fase5", drop_space_prob=0.5), link("d", augment="none", rare_opener_prob=0.15)]
    assert em.recipe_changes(links, [3, 3, 3, 2]) == (
        "b: tanpa jarak antar suku kata; c: argumen sama, langkah tambahan; "
        "d: tanpa augmentasi, 2 font (dari 3), tanpa buang spasi, sisipan aksara langka")
    assert em.recipe_changes(links[:1], [3]) == ""


def test_main_writes_the_card(tmp_path, monkeypatch, capsys):
    card = build(method_repo(tmp_path / "repo"))
    monkeypatch.setattr(em, "build", lambda: card)
    em.main(["--out", str(tmp_path / "hasil")])
    assert json.loads((tmp_path / "hasil/methods.json").read_text(encoding="utf-8")) == json.loads(json.dumps(card))
    out = capsys.readouterr().out
    assert f"run resmi {RUN}" in out and "methods.json" in out and "butir metode" in out

    # Run yang tidak dikenal ekspor hasil: web tidak bisa mencocokkannya, jadi kartunya tidak ditulis.
    unknown = {**card, "official": {**card["official"], "pipeline": None}}
    monkeypatch.setattr(em, "build", lambda: unknown)
    with pytest.raises(SystemExit, match="tidak dikenal scripts/export_results.py"):
        em.main(["--out", str(tmp_path / "ditolak")])
    assert not (tmp_path / "ditolak").exists()

    # Nilai yang bukan JSON ketat (NaN) tidak pernah ditulis: PostgreSQL menolaknya saat impor.
    broken = {**card, "model": {**card["model"], "height": float("nan")}}
    monkeypatch.setattr(em, "build", lambda: broken)
    with pytest.raises(ValueError):
        em.main(["--out", str(tmp_path / "nan")])
    assert not (tmp_path / "nan/methods.json").exists()


REAL = [ROOT / "out/results/manifest.json", ROOT / "data/real/nusaaksara/labels.tsv",
        ROOT / "out/checkpoints" / OFFICIAL_RUN / "last_snapshot.pt"]


@pytest.mark.skipif(not all(p.exists() for p in REAL), reason="checkpoint, data, atau hasil ekspor tidak ada di mesin ini")
def test_real_card_points_at_things_that_exist():
    card = em.build()
    assert card["official"]["run"] == OFFICIAL_RUN and card["chain_complete"] is True
    parameters = card["model"]["parameters"]
    assert parameters["total"] == sum(v for k, v in parameters.items() if k != "total") > 1_000_000
    methods = by_key(card)
    assert len(methods) == sum(len(g["methods"]) for g in card["groups"]) >= 20
    assert {"cnn", "bilstm", "ctc", "greedy", "visual_order", "render", "augment", "optimizer", "cer", "gates"} <= set(methods)
    for m in methods.values():
        assert m["kind"] in em.KINDS and m["status"] in em.STATUSES and len(m["summary"]) > 40
        assert len({pair[0] for pair in m["settings"]}) == len(m["settings"])
        for file in m["files"]:  # berkas sumber yang disebut harus ada di repo
            assert (ROOT / file).exists(), (m["key"], file)
        if m["evidence_source"] and m["evidence_source"].startswith(("out/", "data/", "tests/")):
            assert (ROOT / m["evidence_source"]).exists(), (m["key"], m["evidence_source"])
    # Run resmi ada di rantai, dan tiap run kontrol yang disebut punya checkpoint di mesin ini.
    assert OFFICIAL_RUN in dict(methods["staged"]["settings"])["Rantai run"]
    for _label, value in methods["control"]["settings"]:
        for pair in value.split("; "):
            control = pair.split(" (untuk ")[0]
            assert (ROOT / "out/checkpoints" / control / "last_snapshot.pt").exists(), control
    assert str(card["model"]["classes"]) in methods["head"]["summary"]
    assert "pasti benar" not in json.dumps(card, ensure_ascii=False)

    dump = json.dumps(card, ensure_ascii=False)
    with (ROOT / "data/real/nusaaksara/labels.tsv").open(encoding="utf-8", newline="") as f:
        labels = [line.split("\t")[1] for line in f.read().splitlines()[1:]]
    assert not any(label in dump for label in labels if len(label) >= 4)

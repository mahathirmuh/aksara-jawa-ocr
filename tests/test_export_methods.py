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
from export_results import ALPHA, BETA, OFFICIAL_LINES, OFFICIAL_RUN, WIDTH  # noqa: E402
from src import train  # noqa: E402
from src.augment import PHYSICAL_ORDER  # noqa: E402
from src.dataset import MIN_FRAMES_PER_TARGET, RENDER_SIZE_RANGE, H  # noqa: E402
from src.evaluate import TARGETS  # noqa: E402
from src.model import CRNN  # noqa: E402
from test_export_datasets import SECRET, fake_repo, write_log  # noqa: E402  (repo palsu yang sama dengan kartu data)

ROOT = Path(__file__).resolve().parents[1]
RUN = "fase7_track"  # nama run sungguhan, supaya kunci pipeline dan run kontrolnya dikenal


def test_number_formats_are_indonesian():
    assert em.num(4590909) == "4.590.909" and em.num(93) == "93"
    assert em.dec(0.5) == "0,5" and em.dec(0.0003, 6) == "0,0003" and em.dec(16.0) == "16" and em.dec(0.25) == "0,25"
    assert em.pct(0.2145905) == "21,46%" and em.pct(1.0, 0) == "100%" and em.pct(0.872, 1) == "87,2%"
    assert em.points(-0.0929) == "−9,29 poin" and em.points(0.05) == "+5,00 poin"
    assert em.interval(-0.1192, -0.0683) == "−11,92 sampai −6,83" and em.interval(-0.001, 0.002) == "−0,10 sampai +0,20"
    assert [em.listing(items) for items in ([], ["a"], ["a", "b"], ["a", "b", "c"])] == ["", "a", "a dan b", "a, b, dan c"]


def test_entry_rejects_unknown_kind_and_status():
    assert em.entry("a", "A", "model", "official", "x")["settings"] == []
    assert em.entry("a", "A", "model", "official", "x", evidence=None, evidence_source="berkas")["evidence_source"] is None
    with pytest.raises(ValueError):
        em.entry("a", "A", "ajaib", "official", "x")
    with pytest.raises(ValueError):
        em.entry("a", "A", "model", "rahasia", "x")


def test_training_statements_follow_the_training_source(monkeypatch):
    code = em.train_code()
    assert code == {"optimizer": "AdamW", "scheduler": "OneCycleLR", "pct_start": 0.1, "loss": "CTC"}
    source = inspect.getsource(train.main)
    assert all(text in source for text in em.TRAIN_CODE.values())
    # Kode training berubah (mis. pengoptimal diganti): ekspor harus berhenti, bukan menulis kalimat lama.
    monkeypatch.setitem(em.TRAIN_CODE, "optimizer", "torch.optim.SGD(")
    with pytest.raises(SystemExit, match="tidak lagi cocok dengan kartu metode \\(optimizer\\)"):
        em.train_code()


def tiny_checkpoint(path: Path, classes: int = 6) -> CRNN:
    model = CRNN(classes, channels=(2, 2, 2, 2), hidden=4)
    saved = torch.load(path, weights_only=False) if path.exists() else {"step": 1, "args": {}}
    saved["args"].update({"lr": 3e-4, "weight_decay": 1e-4, "clip": 5.0, "max_batch_columns": 64000, "packed": True})
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**saved, "model": model.state_dict(),
                "model_config": {"n_classes": classes, "channels": [2, 2, 2, 2], "hidden": 4}}, path)
    return model


def test_model_card_is_counted_from_the_built_model(tmp_path):
    path = tmp_path / "run" / "last_snapshot.pt"
    model = tiny_checkpoint(path)
    card = em.model_card(path)
    total = sum(p.numel() for p in model.parameters())
    assert card["parameters"]["total"] == total == sum(v for k, v in card["parameters"].items() if k != "total")
    assert (card["classes"], card["channels"], card["hidden"]) == (6, [2, 2, 2, 2], 4)
    assert (card["conv_layers"], card["kernel"], card["lstm_layers"], card["bidirectional"]) == (7, 3, 2, True)
    assert (card["height"], card["feature_height"], card["width_stride"]) == (96, 3, 4)

    # model_config yang tidak cocok dengan bobotnya ditolak (bobot dimuat ketat).
    saved = torch.load(path, weights_only=False)
    saved["model_config"]["hidden"] = 8
    torch.save(saved, path)
    with pytest.raises(RuntimeError):
        em.model_card(path)


def compare_file(root: Path, a: str, b: str, cer_a: float, cer_b: float, ci: list[float], **extra) -> None:
    folder = root / "out/compare"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{a}_vs_{b}.json").write_text(json.dumps({
        "a": {"key": a}, "b": {"key": b}, "lines": 3,
        "cer": {"a": cer_a, "b": cer_b, "diff": cer_a - cer_b, "bootstrap_clusters": {"ci95": ci},
                "bootstrap_lines": {"ci95": [edge / 2 for edge in ci]}}, **extra}), encoding="utf-8")


def method_repo(root: Path) -> Path:
    """Repo palsu kartu data + yang dibutuhkan kartu metode: bobot model kecil, manifest hasil, berkas pembanding."""
    fake_repo(root, run=RUN)
    model = None
    for run in ("runA", RUN):
        model = tiny_checkpoint(root / "out/checkpoints" / run / "last_snapshot.pt")
    params = sum(p.numel() for p in model.parameters())
    write_log(root / "out/checkpoints" / RUN / "log.jsonl", [
        {"event": "start", "step": 0, "args": {}, "params": params, "device": "xpu"},
        {"step": 10, "val_cer": 0.05, "val_lines": 10}, {"step": 10, "event": "end"}])

    tokenizer = json.loads((root / "data/tokenizer.json").read_text(encoding="utf-8"))
    tokenizer["stats"] = {"lines": 2200, "fonts": ["a.ttf", "b.ttf", "c.ttf"], "prebase": ["U+A9BA", "U+A9BB"],
                          "before_base_rate": {"U+A9BB dirga mure": 1.0, "U+A9BA taling": 1.0, "U+A9BF cakra": 0.03},
                          "reorder_patterns": 370, "non_identity_patterns": 82, "harfbuzz_agreement": 0.99937}
    (root / "data/tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")

    def pipeline(key: str, status: str = "done") -> dict:
        return {"key": key, "label": key.upper(), "kind": "crnn", "status": status, "config": f"konfigurasi {key}"}

    def metric(scope: str, key: str, cer: float, **extra) -> dict:
        return {"scope": scope, "pipeline": key, "lines": 50 if scope == "blind_50" else 3, "cer": cer, **extra}

    (root / "out/results").mkdir(parents=True)
    (root / "out/results/manifest.json").write_text(json.dumps({
        "schema": 1, "generated": "2026-10-04T15:46:51", "limited": False, "official": "crnn_fase7_track",
        "pipelines": [pipeline(k) for k in ("crnn_core", "crnn_fonts", "crnn_fonts_beam", "crnn_fase6_ctrl",
                                            "crnn_fase7_track", "crnn_fase7_ctrl", "vlm_zeroshot")]
        + [pipeline("vlm_finetune", "planned"), pipeline("gpt", "planned")],
        "gates": [{"code": "G1", "value": 0.0027, "target": 0.02, "passed": True},
                  {"code": "G2", "value": 0.012, "target": 0.05, "passed": True},
                  {"code": "G3", "value": 0.20, "target": 0.08, "passed": False},
                  {"code": "G4", "value": 1.0, "target": 1.0, "passed": True}],
        "metrics": [metric("nusaaksara_745", "crnn_core", 0.60), metric("nusaaksara_745", "crnn_fonts", 0.40),
                    metric("nusaaksara_745", "crnn_fonts_beam", 0.35, better=5, worse=2),
                    metric("nusaaksara_745", "crnn_fase7_track", 0.20), metric("nusaaksara_745", "crnn_fase7_ctrl", 0.30),
                    metric("blind_50", "vlm_zeroshot", 0.90), metric("blind_50", "crnn_fonts", 0.40)],
        "ablation": [{"k": 0, "added": "(kontrol)", "G3": 0.75}, {"k": 1, "added": "rotate", "G3": 0.80},
                     {"k": 2, "added": "blur", "G3": 0.75}, {"k": 3, "added": "tight", "G3": 0.55}],
    }), encoding="utf-8")
    # Perlakuan - kontrol, dan arah sebaliknya (kontrol dulu) untuk pasangan font.
    compare_file(root, "crnn_fase7_track", "crnn_fase7_ctrl", 0.20, 0.30, [-0.12, -0.07])
    compare_file(root, "crnn_core", "crnn_fonts", 0.60, 0.40, [0.15, 0.25])
    return root


def by_key(card: dict) -> dict:
    return {m["key"]: m for group in card["groups"] for m in group["methods"]}


def test_build_lists_methods_with_computed_settings_and_evidence(tmp_path):
    root = method_repo(tmp_path)
    card = em.build(root, RUN)
    dump = json.dumps(card, ensure_ascii=False)
    assert SECRET not in dump  # teks label data nyata tidak pernah ikut
    assert card["schema"] == em.SCHEMA and card["results_generated"] == "2026-10-04T15:46:51"
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
        assert (m["evidence"] is None) == (m["evidence_source"] is None)

    params = card["model"]["parameters"]
    assert params["total"] == sum(v for k, v in params.items() if k != "total")
    # Pengaturan tiap butir, lengkap: dari model kecil di checkpoint (kanal 2, tersembunyi 4, 6 kelas), konstanta kode,
    # dan argumen training repo palsu. Satu angka yang pindah butir atau tertukar harus menggagalkan test ini.
    assert methods["cnn"]["settings"] == [
        ["Lapis konvolusi", "7 (3×3, BatchNorm, ReLU)"], ["Kanal", "2, 2, 2, 2"],
        ["Parameter", f"{em.num(params['cnn'] + params['proj'])} (termasuk proyeksi ke 4 dimensi)"]]
    assert f"Tinggi citra {H} piksel dipampatkan menjadi 3, sedangkan lebarnya hanya diperkecil 4 kali" in methods["cnn"]["summary"]
    assert methods["bilstm"]["settings"] == [["Lapis", "2"], ["Unit per arah", "4"], ["Parameter", em.num(params["rnn"])]]
    assert methods["head"]["settings"] == [["Kelas", "6"], ["Parameter", em.num(params["head"])]]
    assert "4 karakter aksara Jawa" in methods["head"]["summary"] and "spasi, dan satu kelas kosong" in methods["head"]["summary"]
    assert len({params["cnn"], params["proj"], params["rnn"], params["head"]}) == 4  # supaya angka yang tertukar terlihat
    assert methods["ctc"]["settings"] == [["Kelas kosong", "indeks 0"],
                                          ["Kolom per karakter, paling sedikit", em.dec(MIN_FRAMES_PER_TARGET)]]
    assert MIN_FRAMES_PER_TARGET != 1 and RENDER_SIZE_RANGE[0] < RENDER_SIZE_RANGE[1]
    assert methods["render"]["settings"] == [
        ["Ukuran huruf", f"{RENDER_SIZE_RANGE[0]} sampai {RENDER_SIZE_RANGE[1]} piksel, diundi per baris"],
        ["Tinggi citra akhir", f"{H} piksel"]]
    assert methods["greedy"]["settings"] == [] and methods["contrast"]["settings"] == []
    assert ["Tanda yang dipindah ke depan", "taling (U+A9BA), dirga mure (U+A9BB)"] in methods["visual_order"]["settings"]
    assert "pada 3 font" in methods["visual_order"]["summary"]
    assert methods["visual_order"]["evidence"] == ("Bolak-balik urutan visual dan urutan Unicode benar pada 2.200 baris "
                                                   "korpus (100%, gerbang G4).")

    assert methods["fonts"]["settings"] == [["Font latih", "3 (2 inti, 1 tambahan)"]]
    assert methods["fonts"]["evidence"] == ("Dari 2 font ke 3 font: G3 60,00% → 40,00% (−20,00 poin; selang kepercayaan "
                                            "95% per halaman −25,00 sampai −15,00).")
    assert methods["fonts"]["evidence_source"] == "out/compare/crnn_core_vs_crnn_fonts.json"
    assert ["Operasi", f"{len(PHYSICAL_ORDER)}: {', '.join(PHYSICAL_ORDER)}"] in methods["augment"]["settings"]
    assert set(em.AUGMENT_WORDS) == set(PHYSICAL_ORDER)  # operasi baru di src.augment wajib diberi kata di kartu
    assert methods["augment"]["summary"] == (
        "Citra bersih hasil render dirusak secara acak supaya mirip pindaian dan foto: diberi tepi kosong, dipotong rapat "
        "sampai tinta menyentuh tepi, dimiringkan sedikit, dilengkungkan halus, ditebal-tipiskan goresannya, diubah "
        "kontrasnya, diberi bayangan dan butiran kertas, diburamkan, diberi bising, dijadikan hitam-putih, diberi bintik, "
        "dan dikompresi. Tiap operasi diundi sendiri-sendiri, jadi tiap citra mendapat campuran yang berbeda.")
    assert methods["augment"]["evidence"] == ("Ablasi 4 run: G3 75,00% → 55,00%. Penurunan terbesar saat menambahkan "
                                              "tight (−20,00 poin), blur (−5,00 poin).")
    assert methods["tracking"]["settings"] == [["Peluang", "0,5"], ["Jarak tambahan", "0 sampai 0,3 em"]]
    assert methods["tracking"]["evidence"] == ("Terhadap run kontrol fase7_ctrl (jumlah langkah sama, tanpa jarak): G3 "
                                               "30,00% → 20,00% (−10,00 poin; selang kepercayaan 95% per halaman −12,00 "
                                               "sampai −7,00).")
    assert methods["drop_space"]["evidence"] == "2 dari 3 baris cetak nyata (66,7%) ditulis tanpa spasi."
    assert "rare" not in methods  # tidak dipakai run resmi dan tidak ada berkas pembandingnya

    assert methods["optimizer"]["settings"] == [["Laju belajar puncak", "0,0003"], ["Peluruhan bobot", "0,0001"],
                                                ["Ukuran batch", "8"], ["Langkah run resmi", "10"]]
    assert methods["onecycle"]["settings"] == [["Bagian naik", "10%"], ["Puncak", "0,0003"]]
    assert methods["clip"]["settings"] == [["Batas norma", "5"]]
    assert methods["batching"]["settings"] == [["Pengemasan LSTM", "ya"], ["Batas kolom per forward", "64.000"]]
    staged = dict(methods["staged"]["settings"])
    assert staged["Rantai run"] == f"runA → {RUN}" and staged["Jumlah langkah"] == "16"
    assert staged["Yang berubah"] == f"{RUN}: augmentasi fase5, 1 font tambahan, buang spasi, jarak antar suku kata"
    assert staged["Perangkat"] == "grafis bawaan Intel (PyTorch XPU)"
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
        ["G1, render bersih", f"CER < {em.pct(TARGETS['G1'], 0)} pada {em.num(OFFICIAL_LINES)} baris uji"],
        ["G2, render rusak berat", f"CER < {em.pct(TARGETS['G2'], 0)} pada {em.num(OFFICIAL_LINES)} baris uji"],
        ["G3, cetakan nyata", f"CER < {em.pct(TARGETS['G3'], 0)} pada 3 baris"], ["G4, tokenizer", "bolak-balik 100%"]]
    # Jumlah pengambilan ulang = bawaan scripts/compare_runs.py, yang membuat berkas pembandingnya.
    resamples = inspect.signature(compare_runs.compare).parameters["resamples"].default
    assert methods["bootstrap"]["settings"] == [["Pengambilan ulang", em.num(resamples)], ["Selang", "persentil 2,5 sampai 97,5"],
                                                ["Satuan", "baris, dan halaman (2 halaman)"]]
    assert f"secara acak {em.num(resamples)} kali" in methods["bootstrap"]["summary"] and resamples != 1000
    assert "bukan keragaman antar-run" in methods["bootstrap"]["summary"]
    assert methods["cer"]["settings"] == [["Jarak", "Levenshtein per karakter"]]
    assert methods["control"]["settings"] == [["Run kontrol", "fase6_ctrl, fase7_ctrl"]]
    assert methods["ablation"]["settings"] == [["Run", "4"]] and "probe" not in methods
    assert methods["vlm_blind"]["status"] == "comparator"
    # "official" = bagian dari model resmi; cara mengukur berstatus "used".
    assert {m["status"] for m in card["groups"][-1]["methods"]} == {"used", "comparator"}
    assert all(m["status"] == "official" for g in card["groups"][:3] for m in g["methods"])
    assert methods["vlm_blind"]["evidence"] == "CER 90,00% melawan 40,00% untuk CRNN saat itu (fase5_fonts)."


def test_optional_evidence_appears_only_when_its_files_exist(tmp_path):
    root = method_repo(tmp_path)
    compare_file(root, "crnn_fase7_track_rare", "crnn_fase7_track", 0.21, 0.20, [-0.01, 0.02], rare={"groups": {"langka": {
        "recall": {"a": 0.5, "b": 0.0}, "precision": {"a": 0.4}}}})
    (root / "out/compare/spacing_synthetic.json").write_text(json.dumps(
        {"settings": {"tracking": [0.0, 0.1, 0.45], "lines": 301}}), encoding="utf-8")
    (root / "out/checkpoints/ablation_0").mkdir(parents=True)
    write_log(root / "out/checkpoints/ablation_0/log.jsonl", [{"event": "start", "step": 0, "args": {"steps": 600}}])
    # Berkas pembanding yang isinya untuk pasangan lain (nama berkasnya saja yang cocok) tidak dipakai.
    compare_file(root, "crnn_fase7_track", "crnn_fase7_ctrl", 0.20, 0.30, [-0.12, -0.07])
    wrong = root / "out/compare/crnn_fase7_track_vs_crnn_fase7_ctrl.json"
    wrong.write_text(wrong.read_text(encoding="utf-8").replace('"crnn_fase7_ctrl"', '"crnn_lain"'), encoding="utf-8")
    methods = by_key(em.build(root, RUN))

    rare = methods["rare"]
    assert rare["status"] == "tested" and rare["settings"] == []
    assert rare["evidence"] == ("Pada 3 baris nyata, aksara langka yang terbaca benar naik dari 0,0% ke 50,0%, tetapi hanya "
                                "40,0% dari keluarannya benar, dan G3 tidak berbeda nyata (21,00% melawan 20,00%). Karena "
                                "itu tidak dipakai di run resmi.")
    # Selang kepercayaan yang tidak memuat nol: selisih G3 disebut apa adanya, bukan "tidak berbeda nyata".
    compare_file(root, "crnn_fase7_track_rare", "crnn_fase7_track", 0.26, 0.20, [0.03, 0.09], rare={"groups": {"langka": {
        "recall": {"a": 0.5, "b": 0.0}, "precision": {"a": 0.4}}}})
    assert "dan G3 berubah +6,00 poin (26,00% melawan 20,00%)" in by_key(em.build(root, RUN))["rare"]["evidence"]
    assert methods["probe"]["settings"] == [["Jarak yang diuji", "0; 0,1; 0,45 em"], ["Baris", "301"]]
    assert methods["probe"]["evidence"] is None  # laporan tanpa hasil per run
    # Baris yang gagal dirender dibuang evaluasinya: yang disebut adalah jumlah yang benar-benar dibaca. Buktinya
    # spasi palsu pada jarak terbesar yang masih di rentang latih run resmi (--track-max 0,3), lawan run kontrolnya.
    def rows(rates):
        return [{"tracking": tracking, "false_per_boundary": rate} for tracking, rate in rates]

    spacing = {"settings": {"tracking": [0.0, 0.15, 0.3, 0.6], "lines": 301}, "fonts": {"javatext.ttf": {"lines": 300, "results": {
        RUN: {"tanpa-spasi": rows([(0.0, 0.0), (0.15, 0.001), (0.3, 0.012), (0.6, 0.675)]), "berspasi": rows([(0.3, 0.5)])},
        "fase7_ctrl": {"tanpa-spasi": rows([(0.0, 0.0), (0.15, 0.6), (0.3, 0.925), (0.6, 0.876)])}}}}}
    (root / "out/compare/spacing_synthetic.json").write_text(json.dumps(spacing), encoding="utf-8")
    probe = by_key(em.build(root, RUN))["probe"]
    assert probe["settings"] == [["Jarak yang diuji", "0; 0,15; 0,3; 0,6 em"], ["Baris", "300 (dari 301 yang diambil)"]]
    assert probe["evidence"] == ("Baris tanpa spasi yang direnggangkan 0,3 em: 1,2 spasi palsu per 100 batas suku kata pada "
                                 "model resmi, 92,5 pada run kontrol fase7_ctrl.")
    assert probe["evidence_source"] == "out/compare/spacing_synthetic.json"
    assert em.spacing_note(spacing, RUN, "fase7_ctrl", 0.15) == (
        "Baris tanpa spasi yang direnggangkan 0,15 em: 0,1 spasi palsu per 100 batas suku kata pada model resmi, 60 pada "
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
    assert methods["ablation"]["settings"] == [["Run", "4"], ["Langkah per run", "600"]]
    # Tanpa berkas pembanding yang sah, selisihnya tetap dari manifest, hanya tanpa selang kepercayaan.
    assert methods["tracking"]["evidence"] == ("Terhadap run kontrol fase7_ctrl (jumlah langkah sama, tanpa jarak): G3 "
                                               "30,00% → 20,00% (−10,00 poin).")
    assert methods["tracking"]["evidence_source"] == "out/results/manifest.json"


def test_augmentation_sentence_names_only_the_operations_of_the_preset(tmp_path):
    root = method_repo(tmp_path)
    evidence = em.Evidence(root)
    real = em.nusaaksara_card(root / "data/real/nusaaksara/labels.tsv")

    def augment(spec: str) -> dict | None:
        group = em.data_group(RUN, {"augment": spec}, {"train": 2, "core": 2, "extra": 0}, real, evidence)
        return {m["key"]: m for m in group["methods"]}.get("augment")

    two = augment("noise+rotate")
    assert two["summary"] == ("Citra bersih hasil render dirusak secara acak supaya mirip pindaian dan foto: dimiringkan "
                              "sedikit dan diberi bising. Tiap operasi diundi sendiri-sendiri, jadi tiap citra mendapat "
                              "campuran yang berbeda.")
    assert ["Operasi", "2: rotate, noise"] in two["settings"] and ["Peluang tiap operasi", "0,5"] in two["settings"]
    # Preset G2: delapan operasi, semuanya selalu dikenakan (p = 1), tanpa operasi pindaian seperti potong rapat.
    heavy = augment("heavy")
    assert heavy["summary"].endswith("diberi bising, dan dikompresi. Semua operasi dikenakan pada tiap citra.")
    assert "dipotong rapat" not in heavy["summary"] and ["Peluang tiap operasi", "1"] in heavy["settings"]
    assert augment("none") is None

    # Run tanpa font tambahan: tidak ada klaim "dari 2 font ke N font", walau berkas pembandingnya ada.
    plain = {m["key"]: m for m in em.data_group(RUN, {"augment": "none"}, {"train": 2, "core": 2, "extra": 0}, real, evidence)["methods"]}
    assert set(plain) == {"render", "fonts"}  # tanpa augmentasi, jarak, buang spasi, dan sisipan
    assert plain["fonts"]["settings"] == [["Font latih", "2 (2 inti, 0 tambahan)"]]
    assert plain["fonts"]["evidence"] is None and plain["fonts"]["evidence_source"] is None

    # Run resmi yang memakai sisipan aksara langka: butirnya bagian dari model resmi dengan peluangnya, dan kalimat
    # "diuji lalu tidak dipakai" tidak boleh muncul walau berkas pembanding uji itu ada.
    compare_file(root, "crnn_fase7_track_rare", "crnn_fase7_track", 0.21, 0.20, [-0.01, 0.02], rare={"groups": {"langka": {
        "recall": {"a": 0.5, "b": 0.0}, "precision": {"a": 0.4}}}})
    used = em.data_group("fase7_track_rare", {"augment": "none", "rare_insert_prob": 0.3, "rare_opener_prob": 0.15},
                         {"train": 2, "core": 2, "extra": 0}, real, em.Evidence(root))
    rare = {m["key"]: m for m in used["methods"]}["rare"]
    assert (rare["status"], rare["evidence"], rare["evidence_source"]) == ("official", None, None)
    assert rare["settings"] == [["Peluang sisip", "0,3"], ["Peluang pembuka", "0,15"]]


def test_gates_and_beam_evidence_follow_what_the_results_contain(tmp_path):
    root = method_repo(tmp_path)
    path = root / "out/results/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    tokenizer = json.loads((root / "data/tokenizer.json").read_text(encoding="utf-8"))
    real = em.nusaaksara_card(root / "data/real/nusaaksara/labels.tsv")
    model = em.model_card(root / "out/checkpoints" / RUN / "last_snapshot.pt")

    # Hasil tanpa gerbang G4 (ekspor lama): tokenizer tidak diberi bukti, dan G4 tidak disebut di daftar gerbang.
    path.write_text(json.dumps({**manifest, "gates": manifest["gates"][:3]}), encoding="utf-8")
    evidence = em.Evidence(root)
    order = {m["key"]: m for m in em.ocr_group(model, tokenizer, evidence)["methods"]}["visual_order"]
    assert order["evidence"] is None and order["evidence_source"] is None
    gates = {m["key"]: m for m in em.evaluation_group(RUN, {}, real, evidence, root)["methods"]}["gates"]
    assert gates["evidence"] == "G1 0,27% · G2 1,20% · G3 20,00% (belum tercapai)."

    # Beam + LM sudah diukur pada checkpoint resmi: angka itu yang dikutip, bukan angka fase5_fonts.
    extra = [{"scope": "nusaaksara_745", "pipeline": "crnn_fase7_track_beam", "lines": 3, "cer": 0.18, "better": 2, "worse": 1}]
    path.write_text(json.dumps({**manifest, "metrics": manifest["metrics"] + extra}), encoding="utf-8")
    beam = em.decoding_group("crnn_fase7_track", em.Evidence(root), root)["methods"][0]
    assert beam["evidence"] == "Pada checkpoint resmi: G3 20,00% → 18,00% (2 baris membaik, 1 memburuk)."
    # Pipeline resmi = fase5_fonts sendiri: hasilnya memang milik checkpoint resmi.
    path.write_text(json.dumps(manifest), encoding="utf-8")
    own = em.decoding_group("crnn_fonts", em.Evidence(root), root)["methods"][0]
    assert own["evidence"] == "Pada checkpoint resmi: G3 40,00% → 35,00% (5 baris membaik, 2 memburuk)."
    # Tidak ada hasil beam sama sekali: butirnya tetap ada, tanpa bukti.
    path.write_text(json.dumps({**manifest, "metrics": [m for m in manifest["metrics"] if not m["pipeline"].endswith("_beam")]}),
                    encoding="utf-8")
    none = em.decoding_group("crnn_fase7_track", em.Evidence(root), root)["methods"][0]
    assert (none["evidence"], none["evidence_source"], none["status"]) == (None, None, "available")


def test_build_stops_when_results_do_not_belong_to_the_run(tmp_path):
    root = method_repo(tmp_path)
    path = root / "out/results/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))

    path.write_text(json.dumps({**manifest, "official": "crnn_fonts"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="menetapkan crnn_fonts sebagai resmi, bukan crnn_fase7_track"):
        em.build(root, RUN)
    path.write_text(json.dumps({**manifest, "limited": True}), encoding="utf-8")
    with pytest.raises(SystemExit, match="--limit"):
        em.build(root, RUN)
    path.unlink()
    with pytest.raises(SystemExit, match="export_results.py"):
        em.build(root, RUN)
    path.write_text(json.dumps(manifest), encoding="utf-8")

    # Tokenizer yang jumlah karakternya tidak cocok dengan lapisan keluaran checkpoint: bukan pasangan yang sama.
    tokenizer_path = root / "data/tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer_path.write_text(json.dumps({**tokenizer, "charset": tokenizer["charset"][:-1]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="jumlah kelas checkpoint berbeda"):
        em.build(root, RUN)
    tokenizer_path.write_text(json.dumps(tokenizer), encoding="utf-8")

    # Jumlah parameter di log training berbeda dari model yang dibangun: checkpoint dan log bukan run yang sama.
    write_log(root / "out/checkpoints" / RUN / "log.jsonl", [{"event": "start", "step": 0, "args": {}, "params": 7, "device": "cpu"}])
    with pytest.raises(SystemExit, match="mencatat 7 parameter"):
        em.build(root, RUN)


def test_training_sentences_follow_the_chain_and_the_logged_device(tmp_path):
    root = method_repo(tmp_path)
    links, _ = em.chain(RUN, root)
    code = em.train_code()
    # Satu run tanpa leluhur, dilatih di CPU: tidak boleh disebut "bertahap" atau "tanpa kartu grafis NVIDIA".
    alone = em.training_group(links[-1:], code, "cpu", root)
    assert alone["intro"].startswith("Model dilatih dalam satu run di CPU, tanpa kartu grafis. ")
    staged = {m["key"]: m for m in alone["methods"]}["staged"]
    assert staged["summary"] == "Model resmi dilatih dalam satu run dari bobot acak."
    assert dict(staged["settings"]) == {"Rantai run": RUN, "Jumlah langkah": "10", "Perangkat": "CPU"}
    # Perangkat lain atau tidak tercatat: kalimatnya tidak menyebut perangkat sama sekali.
    assert em.training_group(links, code, "cuda", root)["intro"].startswith("Model dilatih bertahap. ")
    unknown = em.training_group(links, code, None, root)
    assert unknown["intro"].startswith("Model dilatih bertahap. ")
    assert dict({m["key"]: m for m in unknown["methods"]}["staged"]["settings"])["Perangkat"] == "tidak tercatat"
    # Run eksperimen --no-pack: kartu menyebut LSTM tidak dikemas.
    loose = [{**link, "args": {**link["args"], "packed": False}} for link in links]
    batching = {m["key"]: m for m in em.training_group(loose, code, "xpu", root)["methods"]}["batching"]
    assert batching["settings"][0] == ["Pengemasan LSTM", "tidak"]


def test_main_writes_the_card(tmp_path, monkeypatch, capsys):
    card = em.build(method_repo(tmp_path / "repo"), RUN)
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
    assert card["official"]["run"] == OFFICIAL_RUN
    parameters = card["model"]["parameters"]
    assert parameters["total"] == sum(v for k, v in parameters.items() if k != "total") > 1_000_000
    methods = by_key(card)
    assert len(methods) == sum(len(g["methods"]) for g in card["groups"]) >= 20
    assert {"cnn", "bilstm", "ctc", "greedy", "visual_order", "render", "augment", "optimizer", "cer", "gates"} <= set(methods)
    for m in methods.values():
        assert m["kind"] in em.KINDS and m["status"] in em.STATUSES and len(m["summary"]) > 40
        for file in m["files"]:  # berkas sumber yang disebut harus ada di repo
            assert (ROOT / file).exists(), (m["key"], file)
        if m["evidence_source"] and m["evidence_source"].startswith(("out/", "data/")):
            assert (ROOT / m["evidence_source"]).exists(), (m["key"], m["evidence_source"])
    # Run resmi dan tiap run kontrol yang disebut memang ada di rantai atau di hasil.
    assert OFFICIAL_RUN in dict(methods["staged"]["settings"])["Rantai run"]
    assert str(card["model"]["classes"]) in methods["head"]["summary"]

    dump = json.dumps(card, ensure_ascii=False)
    with (ROOT / "data/real/nusaaksara/labels.tsv").open(encoding="utf-8", newline="") as f:
        labels = [line.split("\t")[1] for line in f.read().splitlines()[1:]]
    assert not any(label in dump for label in labels if len(label) >= 4)

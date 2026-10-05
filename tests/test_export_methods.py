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
UNTIED = ("Laporan evaluasi yang ada di mesin ini tidak bisa dicocokkan dengan checkpoint run resmi dan run kontrolnya "
          "yang sekarang, jadi angkanya tidak dikutip.")
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
    # Bukti selalu membawa sumbernya: kotak "Terukur" di halaman menulis "Sumber:" di bawahnya.
    sourced = em.entry("a", "A", "model", "official", "x", evidence="CER 1%.", evidence_source="out/results/manifest.json")
    assert (sourced["evidence"], sourced["evidence_source"]) == ("CER 1%.", "out/results/manifest.json")
    for missing in (None, ""):
        with pytest.raises(ValueError, match="butir a: bukti tanpa sumber"):
            em.entry("a", "A", "model", "official", "x", evidence="CER 1%.", evidence_source=missing)


@pytest.mark.parametrize("name", sorted(em.CODE_FACTS))
def test_statements_about_the_code_follow_the_source(monkeypatch, name):
    code = em.code_facts()
    assert code == {"optimizer": "AdamW", "scheduler": "OneCycleLR", "pct_start": 0.1, "loss": "CTC"}
    sources = {"src.train.main": inspect.getsource(train.main), "src.dataset.to_tensor": inspect.getsource(to_tensor),
               "src.charlm.CharLM.prob": inspect.getsource(charlm.CharLM.prob)}
    where, text = em.CODE_FACTS[name]
    assert text in sources[where]
    # Yang dijaga adalah baris KODE, bukan kata di komentar atau docstring: mengubah rumusnya harus menghentikan ekspor.
    assert not text.lstrip().startswith(("#", '"""')) and "Witten" not in text
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


def sibling(root: Path, run: str, base: str = RUN, **changes) -> Path:
    """Checkpoint run lain dengan argumen run `base` apa adanya kecuali `changes`: run kontrol atau run perlakuan
    sebuah pasangan uji. Begitulah checkpoint sebenarnya: argparse menyimpan semua argumen, juga yang bawaan."""
    saved = torch.load(root / "out/checkpoints" / base / "last_snapshot.pt", weights_only=False)
    path = root / "out/checkpoints" / run / "last_snapshot.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": saved["step"], "args": {**saved["args"], "run": run, **changes}}, path)
    return path


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
    # Run kontrol perlakuan jarak: argumen run resmi apa adanya, hanya --track-prob 0 (--track-max tetap 0,3, seperti
    # checkpoint fase7_ctrl yang sebenarnya).
    sibling(root, "fase7_ctrl", track_prob=0.0)
    # Pasangan uji font tambahan: sama kecuali --extra-fonts.
    for name, extra in (("fase5_fonts", "fonts/extra"), ("fase5_core", "")):
        save_ckpt(root / "out/checkpoints" / name / "last_snapshot.pt", 7, augment="fase5", drop_space_prob=0.5, extra_fonts=extra)

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
        ["Yang berubah", f"{RUN}: augmentasi fase5, 3 font (dari 2), buang spasi, jarak antar suku kata, kumpulan 400 baris "
                         "(dari 200)"],
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
    sibling(root, "fase7_track_rare", rare_insert_prob=0.3, rare_opener_prob=0.15)
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
    assert methods["probe"]["evidence"] is None  # laporan tanpa hasil per run, tanpa nama font perendernya,
    assert methods["probe"]["note"] == UNTIED    # dan tanpa catatan checkpoint yang dibacanya: itu disebut

    # Baris yang gagal dirender dibuang evaluasinya: yang disebut adalah jumlah yang benar-benar dibaca. Buktinya
    # spasi palsu pada jarak terbesar yang masih di rentang latih run resmi (--track-max 0,3) lawan run kontrolnya,
    # lalu pada jarak terbesar di luar rentang itu.
    def rows(rates):
        return [{"tracking": tracking, "false_per_boundary": rate} for tracking, rate in rates]

    spacing = {"settings": {"tracking": [0.0, 0.15, 0.3, 0.45, 0.6], "lines": 301},
               "checkpoints": {RUN: {"step": 10}, "fase7_ctrl": {"step": 10}}, "fonts": {"javatext.ttf": {"lines": 300, "results": {
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
    sibling(root, "fase7_ctrl", track_prob=0.5)
    unverified = by_key(build(root))
    assert unverified["probe"]["evidence"] is None and unverified["tracking"]["evidence"] is None
    assert unverified["probe"]["settings"] == probe["settings"] and unverified["probe"]["note"] == probe["note"]
    sibling(root, "fase7_ctrl", track_prob=0.0)
    assert by_key(build(root))["probe"]["evidence"] == probe["evidence"]

    # N07: laporan yang dibuat dari checkpoint lain (langkah atau sidik berkasnya berbeda, atau tanpa catatan checkpoint)
    # tidak dikutip sebagai "model resmi", dan itu disebut di catatannya.
    path = root / "out/compare/spacing_synthetic.json"
    official = root / "out/checkpoints" / RUN / "last_snapshot.pt"
    for checkpoints in ({RUN: {"step": 3}, "fase7_ctrl": {"step": 10}}, {RUN: {"step": 10}, "fase7_ctrl": {"step": 4}},
                        {RUN: {"step": 10, "sha1": "000000000000"}, "fase7_ctrl": {"step": 10}}, {"fase7_ctrl": {"step": 10}}):
        path.write_text(json.dumps({**spacing, "checkpoints": checkpoints}), encoding="utf-8")
        stale = by_key(build(root))["probe"]
        assert stale["evidence"] is None and stale["note"] == UNTIED + " " + probe["note"], checkpoints
    path.write_text(json.dumps({**spacing, "checkpoints": {RUN: {"step": 10, "sha1": em.file_sha(official)},
                                                             "fase7_ctrl": {"step": 10}}}), encoding="utf-8")
    assert by_key(build(root))["probe"]["evidence"] == probe["evidence"]
    assert len(em.file_sha(official)) == 12 and em.report_matches({}, root, RUN) is False
    assert em.report_matches({"checkpoints": {"tidak_ada": {"step": 1}}}, root, "tidak_ada") is False
    # Font perender evaluasi dibaca dari laporannya sendiri, bukan disamakan dengan font uji gerbang.
    other_font = json.loads(json.dumps(spacing))
    other_font["fonts"] = {"lain.ttf": other_font["fonts"]["javatext.ttf"]}
    path.write_text(json.dumps(other_font), encoding="utf-8")
    assert by_key(build(root))["probe"]["note"] == ("Kemiripan font perender lain dengan font latih tidak bisa dihitung di "
                                                    "mesin ini: berkas fontnya tidak ada.")
    # c10: rentang jarak yang dilatihkan dibaca dari argumen run resmi, bukan angka tetap 0,3.
    saved = torch.load(official, weights_only=False)
    torch.save({**saved, "args": {**saved["args"], "track_max": 0.15}}, official)
    sibling(root, "fase7_ctrl", track_prob=0.0)
    path.write_text(json.dumps(spacing), encoding="utf-8")
    assert by_key(build(root))["probe"]["evidence"].startswith("Baris tanpa spasi yang direnggangkan 0,15 em: 0,1 spasi palsu")


def data_methods(root: Path, run: str, args: dict, fonts: dict | None = None) -> dict:
    """Butir kelompok data untuk sebuah resep, dengan fakta font bawaan: 2 font inti dari folder, tanpa cacat, tanpa
    spasi sempit, tanpa run sebelumnya di rantai."""
    facts = {"run": run, "train": 2, "core": 2, "extra": 0, "source": "folder", "measured": 2, "current": True,
             "defects": [], "narrow": 0, "recorded": {}, "chain": []}
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
    assert plain["render"]["note"] is None
    assert plain["fonts"]["settings"] == [["Font latih", "2 (2 inti, 0 tambahan)"]]
    assert plain["fonts"]["evidence"] is None and plain["fonts"]["evidence_source"] is None
    recorded = {"fase5_core": 2, "fase5_fonts": 10}
    # ... juga bila ekspor hasil mencatat jumlah font kedua run uji itu: run ini sendiri tidak memakai font tambahan.
    assert data_methods(root, RUN, {"augment": "none"}, {"recorded": recorded})["fonts"]["evidence"] is None
    # Run perlakuan font itu sendiri yang resmi: buktinya miliknya, tanpa "bukan model resmi".
    extra = {"train": 3, "extra": 1, "measured": 3, "recorded": recorded}
    own = data_methods(root, "fase5_fonts", {"augment": "none"}, {**extra, "train": 10, "extra": 8, "measured": 10})
    assert own["fonts"]["evidence"].startswith("Pada run fase5_core lawan fase5_fonts (2 lawan 10 font): G3 60,00% → 40,00% ")
    other = data_methods(root, RUN, {"augment": "none"}, extra)
    assert other["fonts"]["evidence"].startswith("Pada run fase5_core lawan fase5_fonts (2 lawan 10 font, bukan model resmi): ")
    # Tanpa catatan jumlah font kedua run (log atau ekspor hasil), kalimat "N lawan M font" tidak dikarang dari folder.
    assert data_methods(root, RUN, {"augment": "none"}, {**extra, "recorded": {}})["fonts"]["evidence"] is None
    assert data_methods(root, RUN, {"augment": "none"}, {**extra, "recorded": {"fase5_core": 2}})["fonts"]["evidence"] is None
    # N05: jumlah di ekspor hasil yang bertentangan dengan argumen checkpoint (run tanpa --extra-fonts tetapi dicatat 3
    # font) tidak dipakai.
    wrong = {"fase5_core": 3, "fase5_fonts": 10}
    assert data_methods(root, RUN, {"augment": "none"}, {**extra, "recorded": wrong})["fonts"]["evidence"] is None
    # N06: pasangan yang bukan perlakuan-kontrol sungguhan tidak dikutip: jumlah langkah berbeda, argumen data atau
    # pelatihan lain ikut berbeda, "kontrol" yang ikut memakai perlakuannya, atau checkpoint yang tidak ada.
    core = root / "out/checkpoints/fase5_core/last_snapshot.pt"
    for step, changes in ((4, {}), (7, {"augment": "none"}), (7, {"lr": 1e-3}), (7, {"extra_fonts": "fonts/extra"}), (7, {"seed": 3})):
        save_ckpt(core, step, **{"augment": "fase5", "drop_space_prob": 0.5, "extra_fonts": "", **changes})
        assert em.control_run(root, "fonts", "fase5_fonts") is None, (step, changes)
        assert data_methods(root, RUN, {"augment": "none"}, extra)["fonts"]["evidence"] is None
    save_ckpt(core, 7, augment="fase5", drop_space_prob=0.5, extra_fonts="")
    assert em.control_run(root, "fonts", "fase5_fonts") == "fase5_core"
    assert em.control_run(root, "fonts", "run_lain") is None and em.control_run(root, "rare", "fase6_rare") is None
    assert em.control_run(root, "tracking", "fase5_fonts") is None  # terdaftar untuk perlakuan lain saja

    # F01: font latih bercacat disebut di butir render, dengan jumlahnya.
    flawed = data_methods(root, RUN, {"augment": "none"}, {"train": 10, "extra": 8, "measured": 10, "defects": ["BasaJan", "Carakan"]})
    assert flawed["render"]["note"] == (
        "2 dari 10 font latih punya cacat yang diketahui (BasaJan dan Carakan): pada font itu sebagian citra tidak sepadan "
        "dengan labelnya, atau pasangannya tidak menumpuk di lingkungan training. Rinciannya di halaman Dataset, bagian "
        "Font per peran.")
    # N01: font bercacat yang hanya dipakai run sebelumnya di rantai ikut disebut, karena model resmi mewarisi bobotnya.
    chain = [{"run": "runA", "defects": ["BasaJan", "Carakan"], "narrow": 0}, {"run": "runB", "defects": ["Carakan"], "narrow": 0}]
    inherited = data_methods(root, RUN, {"augment": "none"}, {"defects": ["Carakan"], "train": 3, "extra": 1, "measured": 3, "chain": chain})
    assert inherited["render"]["note"].startswith("1 dari 3 font latih punya cacat yang diketahui (Carakan): ")
    assert inherited["render"]["note"].endswith(" Run sebelumnya di rantai (runA) dilatih dengan font bercacat yang tidak "
                                                "dipakai run resmi (BasaJan); model resmi mewarisi bobotnya.")
    assert data_methods(root, RUN, {"augment": "none"}, {"chain": chain})["render"]["note"] == (
        "Run sebelumnya di rantai (runA dan runB) dilatih dengan font bercacat yang tidak dipakai run resmi (BasaJan dan "
        "Carakan); model resmi mewarisi bobotnya.")
    # F08 / N05: folder font sudah berubah sejak run dilatih. Jumlah yang tercatat tetap disebut dengan sumbernya, dan
    # ukuran pada folder sekarang diberi keterangan.
    moved = data_methods(root, RUN, {"augment": "none", "drop_space_prob": 0.5},
                         {"train": 10, "extra": 8, "measured": 9, "current": False, "source": "results", "defects": ["Carakan"], "narrow": 3})
    assert moved["fonts"]["summary"].startswith("Tiap baris dirender dengan satu font yang diundi dari 10 font. ")
    assert moved["fonts"]["settings"] == [["Font latih", "10 (menurut ekspor hasil; folder font sekarang berisi 9)"]]
    assert moved["render"]["note"].startswith("1 dari 9 font latih yang ada di folder sekarang (run ini dilatih dengan 10 "
                                              "font) punya cacat yang diketahui (Carakan): ")
    # N10: porsi baris tanpa spasi tidak dihitung dari berkas yang bukan lagi font run itu.
    assert moved["drop_space"]["settings"] == [
        ["Peluang", "0,5"], ["Selalu, pada font berspasi sempit", "3 dari 9 font yang ada di folder sekarang (run ini "
                                                                 "dilatih dengan 10 font)"]]
    assert "spasi selalu dibuang" in moved["drop_space"]["summary"]
    logged = data_methods(root, RUN, {"augment": "none"}, {"train": 10, "extra": 8, "measured": 9, "current": False, "source": "log"})
    assert logged["fonts"]["settings"] == [["Font latih", "10 (tercatat di log run; 9 di antaranya masih ada di folder font)"]]
    # N09: tidak ada cacat yang terlihat, tetapi sebagian berkas font sudah tidak ada: itu disebut, bukan didiamkan.
    assert logged["render"]["note"] == ("Cacat font hanya bisa diperiksa pada 9 dari 10 font latih: sisanya tidak ada lagi di "
                                        "folder font.")
    # F14: spasi juga selalu dibuang pada font berspasi sempit, jadi porsinya lebih besar dari peluangnya:
    # 0,5 + 0,5 x 3/10 = 65%.
    wide = {"train": 10, "extra": 8, "measured": 10, "narrow": 3}
    narrow = data_methods(root, RUN, {"augment": "none", "drop_space_prob": 0.5}, wide)
    assert narrow["drop_space"]["settings"] == [["Peluang", "0,5"], ["Selalu, pada font berspasi sempit", "3 dari 10 font"],
                                                ["Porsi baris tanpa spasi", "sekitar 65%"]]
    assert "spasi selalu dibuang" in narrow["drop_space"]["summary"] and narrow["drop_space"]["note"] is None
    # N04: aturan font berspasi sempit baru ada sesudah fase5_fonts dilatih. Kartu untuk run itu tidak mengaku memakainya,
    # dan kartu run sesudahnya menyebut run di rantainya yang dilatih sebelum aturan itu ada (hanya yang memakai font
    # berspasi sempit).
    assert set(em.RUNS_BEFORE_NARROW_SPACE_RULE) == {"base", "fase5_quick", "fase5_core", "fase5_fonts"}
    early = data_methods(root, "fase5_fonts", {"augment": "none", "drop_space_prob": 0.5}, wide)["drop_space"]
    assert early["settings"] == [["Peluang", "0,5"]] and "spasi selalu dibuang" not in early["summary"]
    assert early["note"] == ("Run ini dilatih sebelum aturan font berspasi sempit ada: pada 3 font, label yang masih berspasi "
                             "memuat spasi yang nyaris tidak tampak di citra.")
    chain = [{"run": "fase5_quick", "defects": [], "narrow": 0}, {"run": "fase5_fonts", "defects": [], "narrow": 3},
             {"run": "fase6_ctrl", "defects": [], "narrow": 3}]
    later = data_methods(root, RUN, {"augment": "none", "drop_space_prob": 0.5}, {**wide, "chain": chain})["drop_space"]
    assert later["settings"][-1] == ["Porsi baris tanpa spasi", "sekitar 65%"]
    assert later["note"] == ("Run fase5_fonts di rantai dilatih sebelum aturan font berspasi sempit ada: di font itu labelnya "
                             "masih memuat spasi yang nyaris tidak tampak di citra.")

    # Run resmi yang memakai sisipan aksara langka: butirnya bagian dari model resmi dengan peluangnya, dan kalimat
    # "diuji lalu tidak dipakai" tidak boleh muncul walau berkas pembanding uji itu ada.
    manifest = manifest_of(root)
    manifest["metrics"].append({"scope": "nusaaksara_745", "pipeline": "crnn_fase7_track_rare", "lines": 3, "cer": 0.21})
    write_manifest(root, manifest)
    compare_file(root, "crnn_fase7_track_rare", "crnn_fase7_track", 0.21, 0.20, [-0.01, 0.02], rare={"groups": {"langka": {
        "recall": {"a": 0.5, "b": 0.0}, "precision": {"a": 0.4}}}})
    sibling(root, "fase7_track_rare", rare_insert_prob=0.3, rare_opener_prob=0.15)
    rare = data_methods(root, "fase7_track_rare", {"augment": "none", "rare_insert_prob": 0.3, "rare_opener_prob": 0.15})["rare"]
    assert (rare["status"], rare["evidence"], rare["evidence_source"]) == ("official", None, None)
    assert rare["settings"] == [["Peluang sisip", "0,3"], ["Peluang pembuka", "0,15"]]
    # Hanya pembuka baris: peluang sisip tetap ditulis (nol), bukan galat.
    opener = data_methods(root, "fase7_track_rare", {"augment": "none", "rare_opener_prob": 0.15})["rare"]
    assert opener["settings"] == [["Peluang sisip", "0"], ["Peluang pembuka", "0,15"]]
    # Run lain yang BUKAN kontrol uji sisipan itu tidak mengutip hasilnya.
    assert "rare" not in data_methods(root, "fase7_ctrl", {"augment": "none"})
    tested = data_methods(root, RUN, {"augment": "none"})["rare"]
    assert tested["status"] == "tested" and tested["evidence"].endswith(
        "tetapi hanya 40,0% dari keluarannya benar, dan G3 tidak berbeda nyata (21,00% melawan 20,00%). Karena itu tidak "
        "dipakai di run resmi.")
    # Run kontrol uji itu sendiri, bila kelak dilatih DENGAN sisipan: butirnya "dipakai", jadi hasil uji yang berakhir
    # "karena itu tidak dipakai di run resmi" tidak dikutip.
    used = data_methods(root, RUN, {"augment": "none", "rare_insert_prob": 0.3})["rare"]
    assert (used["status"], used["evidence"], used["evidence_source"]) == ("official", None, None)
    # N06: run "perlakuan" yang checkpoint-nya ternyata tanpa sisipan bukan pasangan uji, jadi hasilnya tidak dikutip.
    sibling(root, "fase7_track_rare")
    assert em.control_run(root, "rare", "fase7_track_rare") is None and "rare" not in data_methods(root, RUN, {"augment": "none"})
    sibling(root, "fase7_track_rare", rare_insert_prob=0.3, rare_opener_prob=0.15)
    assert em.control_run(root, "rare", "fase7_track_rare") == RUN
    # N10: kalimatnya mengikuti hasilnya. Presisi tinggi tidak ditulis "tetapi hanya", recall yang turun tidak ditulis
    # "naik", dan G3 yang nyata lebih baik tidak diberi "karena itu tidak dipakai".
    manifest["metrics"][-1]["cer"] = 0.14
    write_manifest(root, manifest)
    compare_file(root, "crnn_fase7_track_rare", "crnn_fase7_track", 0.14, 0.20, [-0.09, -0.03], rare={"groups": {"langka": {
        "recall": {"a": 0.5, "b": 0.6}, "precision": {"a": 0.95}}}})
    assert data_methods(root, RUN, {"augment": "none"})["rare"]["evidence"] == (
        "Pada 3 baris nyata, aksara langka yang terbaca benar berubah dari 60,0% ke 50,0%, dan 95,0% dari keluarannya benar, "
        "dan G3 berubah −6,00 poin (14,00% melawan 20,00%). Run resmi tidak memakainya.")
    # G3 yang nyata lebih BURUK tetap berakhir "karena itu tidak dipakai".
    manifest["metrics"][-1]["cer"] = 0.26
    write_manifest(root, manifest)
    compare_file(root, "crnn_fase7_track_rare", "crnn_fase7_track", 0.26, 0.20, [0.03, 0.09], rare={"groups": {"langka": {
        "recall": {"a": 0.5, "b": 0.0}, "precision": {"a": 0.5}}}})
    assert data_methods(root, RUN, {"augment": "none"})["rare"]["evidence"].endswith(
        "naik dari 0,0% ke 50,0%, dan 50,0% dari keluarannya benar, dan G3 berubah +6,00 poin (26,00% melawan 20,00%). "
        "Karena itu tidak dipakai di run resmi.")

    # F06: bukti jarak hanya terhadap run kontrol yang memang dilatih TANPA jarak. fase7_track_rare (jarak + sisipan)
    # tidak punya kontrol jarak sendiri: fase7_track adalah kontrol sisipannya dan dilatih dengan jarak.
    tracked = {"augment": "none", "track_prob": 0.5, "track_max": 0.3}
    assert data_methods(root, RUN, tracked)["tracking"]["evidence"].startswith("Terhadap run kontrol fase7_ctrl (jumlah langkah sama, tanpa jarak): ")
    both = data_methods(root, "fase7_track_rare", {**tracked, "rare_insert_prob": 0.3})["tracking"]
    assert both["evidence"] is None and both["settings"] == [["Peluang", "0,5"], ["Jarak tambahan", "0 sampai 0,3 em"]]
    assert em.control_run(root, "tracking", "fase7_track_rare") is None and em.control_run(root, "tracking", RUN) == "fase7_ctrl"
    # Run kontrol yang ternyata dilatih dengan jarak, dengan jumlah langkah lain, dengan argumen data lain, atau yang
    # checkpoint-nya tidak ada: tidak dipakai (N06: "jumlah langkah sama" hanya ditulis sesudah diperiksa).
    sibling(root, "fase7_ctrl", track_prob=0.5)
    assert em.control_run(root, "tracking", RUN) is None and data_methods(root, RUN, tracked)["tracking"]["evidence"] is None
    control = sibling(root, "fase7_ctrl", track_prob=0.0)
    saved = torch.load(control, weights_only=False)
    torch.save({**saved, "step": 4}, control)
    assert em.control_run(root, "tracking", RUN) is None and data_methods(root, RUN, tracked)["tracking"]["evidence"] is None
    sibling(root, "fase7_ctrl", track_prob=0.0, drop_space_prob=0.0)
    assert em.control_run(root, "tracking", RUN) is None
    # --track-max kontrol tidak berarti apa-apa selama --track-prob-nya 0 (begitulah checkpoint fase7_ctrl sebenarnya).
    sibling(root, "fase7_ctrl", track_prob=0.0, track_max=0.9)
    assert em.control_run(root, "tracking", RUN) == "fase7_ctrl"
    control.unlink()
    assert em.control_run(root, "tracking", RUN) is None


def test_ablation_evidence_does_not_rank_single_steps():
    def rows(*steps):
        return [{"added": name, "G3": value} for name, value in steps]

    # Operasi preset dan tiruan pindaian bercampur urutannya: tiap langkah disebut apa adanya, tanpa dijumlahkan.
    mixed = em.ablation_evidence(rows(("(kontrol)", 0.75), ("tight", 0.60), ("rotate", 0.58), ("stroke", 0.50)), [0, 1, 0, 0])
    assert mixed[0] == ("Ablasi 4 run: G3 75,00% → 50,00%. Selisih tiap langkah: tight −15,00 poin, rotate −2,00 poin, "
                        "stroke −8,00 poin.")
    # Tidak ada langkah yang naik: catatannya tidak menyebut kenaikan. Seed berbeda-beda: tidak disebut sama.
    assert mixed[1] == ("Operasi ditambahkan berurutan dan tiap run dilatih satu kali, jadi selisih satu langkah mencampur "
                        "pengaruh operasinya dengan derau antar-run. G3 adalah data uji: angka ini dibaca sebagai arah, "
                        "bukan dasar memilih operasi.")
    three = rows(("(kontrol)", 0.75), ("rotate", 0.80), ("blur", 0.70))
    only_preset = em.ablation_evidence(three, [0, 0, 0])
    assert only_preset[0] == "Ablasi 3 run: G3 75,00% → 70,00%. Selisih tiap langkah: rotate +5,00 poin, blur −10,00 poin."
    assert "dengan seed yang sama" in only_preset[1] and "(menambahkan rotate malah menaikkan G3 5,00 poin)" in only_preset[1]
    # N10: "seed yang sama" hanya bila log SEMUA run ablasi ada dan seed-nya satu. Log yang hilang, log pertama saja,
    # atau satu run berseed lain: tidak disebut sama.
    for seeds in ([0, None, 0], [0], [0, 0], [0, 0, 1], [1, 0, 0], [None, None, None], []):
        assert "dengan seed yang sama" not in em.ablation_evidence(three, seeds)[1], seeds
    # Satu baris, atau baris tanpa G3: tidak ada bukti.
    assert em.ablation_evidence(rows(("(kontrol)", 0.75)), [0]) is None
    assert em.ablation_evidence(rows(("(kontrol)", 0.75), ("rotate", None)), [0, 0]) is None
    assert em.ablation_evidence([], []) is None


def test_fonts_are_counted_from_the_log_then_the_results_then_the_folder(tmp_path, monkeypatch):
    root = method_repo(tmp_path)
    links, _ = em.chain(RUN, root)
    charset = [" ", KA, GA]
    recorded = em.recorded_fonts(root)
    assert recorded == {"fase5_core": 2, "fase5_fonts": 3, "fase6_ctrl": 3, "fase7_track": 3, "fase7_ctrl": 3}
    core = [font.name for font in TRAIN_FONTS]
    log = links[-1]["path"].parent / "log.jsonl"

    def facts(link=None, **kwargs):
        got = em.font_facts(root, link or links[-1], kwargs.get("charset", charset), kwargs.get("recorded", recorded))
        return got, (got["train"], got["core"], got["extra"], got["source"], got["measured"], got["current"])

    full, numbers = facts()
    assert numbers == (3, 2, 1, "results", 3, True) and full["run"] == RUN
    assert full["defects"] == [] and full["narrow"] == 0
    assert [path.name for path in full["paths"]] == core + ["Salinan.ttf"]
    # Run tanpa catatan di ekspor hasil: dari folder sekarang.
    assert facts(links[0])[1] == (2, 2, 0, "folder", 2, True)
    # N05: jumlah di ekspor hasil yang bertentangan dengan argumen checkpoint tidak dipakai (run tanpa --extra-fonts
    # yang dicatat 3 font), dan folder yang isinya lebih banyak atau lebih sedikit dari catatan bukan "sama".
    assert facts(links[0], recorded={"runA": 3})[1] == (2, 2, 0, "folder", 2, True)
    assert facts(links[0], recorded={"runA": 2})[1] == (2, 2, 0, "results", 2, True)
    assert facts(recorded={RUN: 5})[1] == (5, 2, 3, "results", 3, False)
    assert facts(recorded={RUN: 2})[1] == (2, 2, 0, "results", 3, False)

    # Log run yang mencatat NAMA fontnya didahulukan dari keduanya, dan hanya berkas bernama itu yang diukur.
    write_log(log, [{**start(0), "fonts": core + ["Salinan.ttf", "SudahDihapus.ttf"]}, {"step": 10, "event": "end"}])
    assert facts()[1] == (4, 2, 2, "log", 3, False)
    # N02: jumlahnya sama dengan folder tetapi namanya lain: folder itu BUKAN font run ini.
    write_log(log, [{**start(0), "fonts": core + ["Lain.ttf"]}, {"step": 10, "event": "end"}])
    renamed, numbers = facts()
    assert numbers == (3, 2, 1, "log", 2, False) and [path.name for path in renamed["paths"]] == core
    # Folder berisi LEBIH banyak daripada yang dicatat log (font ditambahkan sesudah run): yang tidak tercatat tidak diukur.
    write_log(log, [{**start(0), "fonts": core}, {"step": 10, "event": "end"}])
    fewer, numbers = facts()
    assert numbers == (2, 2, 0, "log", 2, True) and [path.name for path in fewer["paths"]] == core
    # Run yang dilanjutkan: daftar proses TERAKHIR sebelum langkah checkpoint yang dipakai. Proses yang mulai tepat di
    # langkah checkpoint (10) baru melanjutkannya, jadi daftarnya bukan milik checkpoint ini.
    write_log(log, [{**start(0), "fonts": core}, {**start(5), "fonts": core + ["Salinan.ttf"]},
                    {"step": 10, "event": "end"}, {**start(10), "fonts": core[:1]}])
    assert facts()[1] == (3, 2, 1, "log", 3, True)
    write_log(log, [start(0), {"step": 10, "event": "end"}, {**start(10), "fonts": core[:1]}])
    assert facts()[1] == (3, 2, 1, "results", 3, True)
    # Proses tanpa baris sintetis mencatat daftar kosong: itu bukan "tidak tercatat".
    write_log(log, [{**start(0), "fonts": []}, {"step": 10, "event": "end"}])
    assert facts()[1] == (0, 0, 0, "log", 0, True)
    write_log(log, [{**start(0), "fonts": core + ["Salinan.ttf", "SudahDihapus.ttf"]}, {"step": 10, "event": "end"}])

    # Cacat = font di daftar cacat yang diketahui atau yang tidak punya glyph sebuah karakter charset; spasi sempit =
    # aturan src.dataset (celah kata hasil shaping di bawah ambang).
    monkeypatch.setattr(em, "FONT_DEFECTS", ("Salinan.ttf",))
    monkeypatch.setattr(em, "space_ratio", lambda path: 0.05 if Path(path).name == TRAIN_FONTS[1].name else 0.2)
    measured = facts(charset=charset + ["꧎"])[0]  # U+A9CE: tidak ada di font mana pun
    assert measured["defects"] == sorted(path.stem for path in measured["paths"]) and measured["narrow"] == 1
    assert facts()[0]["defects"] == ["Salinan"]

    # Font uji yang sekeluarga dengan font latih: salinan font inti pertama (73 dari 73 lebar maju sama).
    owners = em.chain_fonts([measured])
    assert {name: owner["runs"] for name, owner in owners.items()} == {name: [RUN] for name in core + ["Salinan.ttf"]}
    twins, unknown = em.family_twins(owners, ["javatext.ttf", "javatext.ttf"], root / "ujifont")
    assert unknown == [] and [(twin["test"], twin["train"], twin["same"], twin["of"], twin["runs"]) for twin in twins] == [
        ("javatext", TRAIN_FONTS[0].stem, 73, 73, [RUN]), ("javatext", "Salinan", 73, 73, [RUN])]
    assert em.family_note(twins, "G1 dan G2", RUN, [], False) == FAMILY_NOTE.format("G1 dan G2")
    # Tidak sekeluarga dengan font latih mana pun: tidak ada catatan.
    assert em.family_twins(em.chain_fonts([{"run": RUN, "paths": [TRAIN_FONTS[1]]}]), ["javatext.ttf"], root / "ujifont") == ([], [])
    assert em.family_note([], "G1 dan G2", RUN, [], False) is None and em.probe_note([], []) is None
    # N09: font uji yang berkasnya tidak ada di mesin ini, atau font latih yang berkasnya sudah tidak ada: kemiripannya
    # tidak bisa dihitung, dan itu DITULIS (dulu catatannya hilang begitu saja).
    assert em.family_twins(owners, ["javatext.ttf"], root / "tidak-ada") == ([], ["javatext"])
    missing = "Kemiripan font uji javatext dengan font latih tidak bisa dihitung di mesin ini: berkas fontnya tidak ada."
    assert em.family_note([], "G1 dan G2", RUN, ["javatext"], False) == missing
    gone = "Sebagian font latih tidak ada lagi di folder font, jadi kemiripannya dengan font uji tidak ikut diperiksa."
    assert em.family_note([], "G1 dan G2", RUN, [], True) == gone
    assert em.family_note(twins, "G1 dan G2", RUN, ["lain"], True) == " ".join([
        FAMILY_NOTE.format("G1 dan G2"), missing.replace("javatext", "lain"), gone])
    assert em.probe_note([], ["javatext"]) == ("Kemiripan font perender javatext dengan font latih tidak bisa dihitung di "
                                               "mesin ini: berkas fontnya tidak ada.")

    # N01: yang "pernah dilihat model" adalah font seluruh rantai. Run resmi yang sendiri hanya memakai font lain tetap
    # mendapat catatan bila run sebelumnya di rantai dilatih dengan font sekeluarga, dan itu disebut milik run mana.
    earlier = {"run": "runA", "paths": [TRAIN_FONTS[0], root / "fonts/extra/Salinan.ttf"]}
    last = {"run": RUN, "paths": [TRAIN_FONTS[1]]}
    owners = em.chain_fonts([earlier, last])
    assert [(name, owner["runs"]) for name, owner in owners.items()] == [
        (TRAIN_FONTS[0].name, ["runA"]), ("Salinan.ttf", ["runA"]), (TRAIN_FONTS[1].name, [RUN])]
    chained, _ = em.family_twins(owners, ["javatext.ttf"], root / "ujifont")
    assert em.family_note(chained, "G1 dan G2", RUN, [], False) == (
        f"Font uji javatext sekeluarga dengan font latih {TRAIN_FONTS[0].stem} (font run runA di rantai) dan Salinan (font "
        "run runA di rantai) (lebar 73 dari 73 aksara, angka, dan pada persis sama), jadi G1 dan G2 mengukur generalisasi di "
        "dalam keluarga huruf itu, bukan ke font yang belum pernah dilihat model.")
    both = em.chain_fonts([earlier, {"run": RUN, "paths": [TRAIN_FONTS[0], TRAIN_FONTS[1]]}])
    assert both[TRAIN_FONTS[0].name]["runs"] == ["runA", RUN]
    shared, _ = em.family_twins(both, ["javatext.ttf"], root / "ujifont")
    assert em.family_note(shared, "G1 dan G2", RUN, [], False).startswith(
        f"Font uji javatext sekeluarga dengan font latih {TRAIN_FONTS[0].stem} dan Salinan (font run runA di rantai) (lebar ")
    # N10: tiap font latih disebut dengan tumpang-tindihnya sendiri, bukan yang terbesar untuk semuanya.
    two = [{"test": "uji", "train": "A", "runs": [RUN], "same": 72, "of": 73}, {"test": "uji", "train": "B", "runs": [RUN], "same": 66, "of": 73}]
    assert em.family_note(two, "G1", RUN, [], False).startswith(
        "Font uji uji sekeluarga dengan font latih A dan B (lebar berturut-turut 72 dan 66 dari 73 aksara, angka, dan pada "
        "persis sama), jadi G1 mengukur ")
    assert em.probe_note(two, []) == ("Citranya dirender dengan font uji uji, yang sekeluarga dengan font latih A dan B, jadi "
                                      "hasilnya belum tentu berlaku untuk bentuk huruf lain.")

    # Dari ujung ke ujung: run resmi yang sendiri tanpa font tambahan, melanjutkan runA yang dilatih dengan font tambahan
    # sekeluarga font uji. Font inti pertama tetap dipakai run resmi; Salinan hanya dipakai runA.
    write_log(log, [start(0), {"step": 10, "event": "end"}])
    saved = torch.load(links[0]["path"], weights_only=False)
    torch.save({**saved, "args": {**saved["args"], "extra_fonts": "fonts/extra"}}, links[0]["path"])
    saved = torch.load(links[-1]["path"], weights_only=False)
    torch.save({**saved, "args": {**saved["args"], "extra_fonts": ""}}, links[-1]["path"])
    manifest = manifest_of(root)
    for pipeline in manifest["pipelines"]:
        pipeline["config"] = pipeline["config"].replace(" · 3 font", "")
    write_manifest(root, manifest)
    gates = by_key(em.build(root, RUN, root / "ujifont"))["gates"]
    assert gates["note"].startswith(f"Font uji javatext sekeluarga dengan font latih {TRAIN_FONTS[0].stem} dan Salinan (font run "
                                    "runA di rantai) (lebar 73 dari 73 ")
    # Font uji tidak ada di mesin ini: catatannya tidak hilang, melainkan menyebut tidak bisa dihitung.
    assert by_key(em.build(root, RUN, root / "tidak-ada"))["gates"]["note"] == missing
    # Font latih yang dicatat log tetapi berkasnya sudah dihapus: kemiripannya tidak ikut diperiksa, dan itu disebut.
    write_log(log, [{**start(0), "fonts": core + ["SudahDihapus.ttf"]}, {"step": 10, "event": "end"}])
    assert by_key(em.build(root, RUN, root / "ujifont"))["gates"]["note"].endswith(" " + gone)


def test_gates_and_beam_evidence_follow_what_the_results_contain(tmp_path):
    root = method_repo(tmp_path)
    manifest = manifest_of(root)
    tokenizer = json.loads((root / "data/tokenizer.json").read_text(encoding="utf-8"))
    real = em.nusaaksara_card(root / "data/real/nusaaksara/labels.tsv")
    model = em.model_card(root / "out/checkpoints" / RUN / "last_snapshot.pt")

    def evaluation(twins=(), fonts=()):
        family = {"owners": {}, "dir": root / "ujifont", "twins": list(twins), "unknown": [], "incomplete": False}
        group = em.evaluation_group(RUN, {"track_max": 0.3}, real, em.Evidence(root), root, family, list(fonts))
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
    # Hasil kedua run ada di manifest, tetapi checkpoint "kontrol"-nya bukan kontrol yang sah (dilatih dengan jarak juga,
    # atau sampai langkah lain): pasangan itu tidak disebut run kontrol. Daftar CONTROLS hanya calon.
    write_manifest(root, manifest)
    both = [["Kontrol font tambahan", "fase5_core (untuk fase5_fonts)"], ["Kontrol jarak antar suku kata", "fase7_ctrl (untuk fase7_track)"]]
    assert evaluation()["control"]["settings"] == both
    sibling(root, "fase7_ctrl")
    assert evaluation()["control"]["settings"] == both[:1]
    ctrl = sibling(root, "fase7_ctrl", track_prob=0.0)
    saved = torch.load(ctrl, weights_only=False)
    torch.save({**saved, "step": saved["step"] + 1}, ctrl)
    assert evaluation()["control"]["settings"] == both[:1]
    sibling(root, "fase7_ctrl", track_prob=0.0)
    assert evaluation()["control"]["settings"] == both

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

    # c06: persen gerbang G4 dibaca dari manifest, bukan diketik.
    low = dict(manifest["gates"][3], value=0.95)
    write_manifest(root, {**manifest, "gates": manifest["gates"][:3] + [low]})
    order = {m["key"]: m for m in em.ocr_group(model, tokenizer, em.Evidence(root))["methods"]}["visual_order"]
    assert order["evidence"].startswith("Gerbang G4, 95%: ")
    write_manifest(root, manifest)

    # Berkas pembanding hanya dipakai bila dihitung dari prediksi yang sama dengan manifest: CER kedua sisinya harus
    # persis sama (c09) dan kedua pipeline-nya harus ada di manifest (c01). Pengaturan bootstrap hanya dikutip dari
    # berkas yang lolos pemeriksaan itu (c05). Selisihnya sendiri selalu dari manifest.
    evidence = em.Evidence(root)
    compare_file(root, "crnn_fase7_track", "crnn_fase7_ctrl", 0.20, 0.30, [-0.12, -0.07])
    assert evidence.compare("crnn_fase7_track", "crnn_fase7_ctrl")["cer"]["a"] == 0.20
    assert evidence.bootstrap()["resamples"] == 2000
    compare_file(root, "crnn_fase7_track", "crnn_fase7_ctrl", 0.2005, 0.30, [-0.12, -0.07])  # beda 0,05 poin: ekspor lain
    assert evidence.compare("crnn_fase7_track", "crnn_fase7_ctrl") is None
    assert evidence.g3_change("crnn_fase7_track", "crnn_fase7_ctrl") == ("G3 30,00% → 20,00% (−10,00 poin)", "out/results/manifest.json")
    compare_file(root, "crnn_x", "crnn_y", 0.10, 0.20, [-0.12, -0.07])  # pipeline yang tidak ada di manifest
    assert evidence.compare("crnn_x", "crnn_y") is None and evidence.compare("crnn_tidak", "crnn_ada") is None
    assert evidence.bootstrap() is None
    assert evaluation()["bootstrap"]["settings"][0] == ["Pengambilan ulang", em.num(resamples)]


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

    # N10: jumlah langkah yang direncanakan dibaca dari jadwal di checkpoint ("planned"), juga untuk run yang dimulai
    # dengan --epochs (args.steps 0) lalu dihentikan lebih awal: run itu TIDAK menyelesaikan siklusnya.
    epochs = [{**links[-1], "args": {**links[-1]["args"], "steps": 0}, "planned": 60}]
    _, methods = training(epochs)
    assert methods["onecycle"]["summary"].endswith("Run resmi dihentikan di langkah 10 dari 60 yang direncanakan, sebelum "
                                                   "siklusnya selesai.")
    assert ["Langkah direncanakan run resmi", "60"] in methods["onecycle"]["settings"]
    _, methods = training([{**links[0], "planned": 60}, links[-1]])
    assert "1 dari 1 run sebelumnya di rantai (runA) dihentikan" in methods["onecycle"]["summary"]
    # Tanpa jadwal di checkpoint dan tanpa --steps: langkah yang dicapai dianggap rencananya.
    _, methods = training([{**links[-1], "args": {**links[-1]["args"], "steps": 0}}])
    assert methods["onecycle"]["summary"].endswith("Run resmi menyelesaikan siklusnya (10 langkah).")
    path = links[-1]["path"]
    assert em.planned_steps(path) is None
    saved = torch.load(path, weights_only=False)
    torch.save({**saved, "scheduler": {"total_steps": 60, "last_epoch": 10}}, path)
    assert em.planned_steps(path) == 60
    assert by_key(build(root))["onecycle"]["summary"].endswith("Run resmi dihentikan di langkah 10 dari 60 yang direncanakan, "
                                                               "sebelum siklusnya selesai.")
    torch.save(saved, path)

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
        "b: tanpa jarak antar suku kata; c: argumen data sama, langkah tambahan; "
        "d: tanpa augmentasi, 2 font (dari 3), tanpa buang spasi, sisipan aksara langka")
    assert em.recipe_changes(links[:1], [3]) == ""
    # N06: nilai yang berubah ikut disebut; "argumen data sama" hanya bila SEMUA argumen data sama.
    before = link("x", augment="fase5", drop_space_prob=0.5, track_prob=0.5, track_max=0.3, train_lines=400,
                  rare_insert_prob=0.3, extra_fonts="fonts/extra")
    after = link("y", augment="fase5", drop_space_prob=1.0, track_prob=0.9, track_max=0.6, train_lines=500, seed=3,
                 rare_insert_prob=0.1, extra_fonts="fonts/lain", real_train="data/real/labels.tsv")
    assert em.recipe_changes([before, after], [3, 3]) == (
        "y: font tambahan dari fonts/lain, buang spasi p 1 (dari 0,5), jarak antar suku kata p 0,9 hingga 0,6 em (dari p 0,5 "
        "hingga 0,3 em), sisipan aksara langka dengan pengaturan lain, kumpulan 500 baris (dari 400), seed 3 (dari 0), "
        "data nyata ikut dipakai")
    for key, value in (("drop_space_prob", 0.6), ("track_max", 0.4), ("rare_insert_prob", 0.2), ("train_lines", 401), ("seed", 1)):
        changed = em.recipe_changes([before, link("y", **{**before["args"], key: value})], [3, 3])
        assert "argumen data sama" not in changed and changed.startswith("y: "), key
    # Argumen pelatihan (laju belajar, jumlah langkah) bukan argumen data, dan pengaturan yang induknya mati (jarak
    # terbesar tanpa peluang jarak, tempelan sisipan tanpa sisipan) tidak berarti apa-apa.
    same = [link("p", augment="fase5", lr=5e-4, steps=1500, track_max=0.3),
            link("q", augment="fase5", lr=3e-4, steps=600, track_max=0.9, rare_attach=True, rare_max_similarity=0.9)]
    assert em.recipe_changes(same, [2, 2]) == "q: argumen data sama, langkah tambahan"
    off = [link("p", augment="fase5", extra_fonts="fonts/extra", real_val="x"), link("q", augment="fase5")]
    assert em.recipe_changes(off, [3, 3]) == "q: tanpa font tambahan, tanpa data nyata"
    assert em.data_recipe({}) == em.data_recipe({"track_max": 0.3, "rare_attach": True, "rare_max_similarity": 0.9, "augment": None})
    assert em.data_recipe({"track_prob": 0.5, "track_max": 0.3})["track_max"] == 0.3
    assert set(em.TREATMENT_KEYS) == set(em.CONTROLS) == set(em.TREATMENTS)
    assert all(key in em.data_recipe({}) for keys in em.TREATMENT_KEYS.values() for key in keys)


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

    # N03: bukti yang berkasnya ada di mesin ini memang muncul di kartu. Test ini dulu tidak memeriksa satu bukti pun,
    # jadi pemeriksaan pasangan kontrol yang salah bisa mengosongkan bukti di kartu sungguhan tanpa ketahuan (repo
    # palsu tidak menirukan semua argumen checkpoint yang sebenarnya).
    compare = ROOT / "out/compare"
    if OFFICIAL_RUN == "fase7_track":
        controls = dict(methods["control"]["settings"])
        assert controls["Kontrol jarak antar suku kata"] == "fase7_ctrl (untuk fase7_track)"
        assert controls["Kontrol font tambahan"] == "fase5_core (untuk fase5_fonts)"
        assert methods["tracking"]["evidence"].startswith("Terhadap run kontrol fase7_ctrl (jumlah langkah sama, tanpa jarak): G3 ")
        assert methods["fonts"]["evidence"].startswith("Pada run fase5_core lawan fase5_fonts (2 lawan 10 font, bukan model resmi): G3 ")
        if (compare / "crnn_fase7_track_vs_crnn_fase7_ctrl.json").exists():
            assert "selang kepercayaan 95% per halaman" in methods["tracking"]["evidence"]
        if (compare / "crnn_fase7_track_rare_vs_crnn_fase7_track.json").exists():
            assert controls["Kontrol sisipan aksara langka"].startswith("fase7_track (untuk fase7_track_rare)")
            assert methods["rare"]["status"] == "tested" and "dari keluarannya benar" in methods["rare"]["evidence"]
        if (compare / "spacing_synthetic.json").exists():
            assert methods["probe"]["evidence"].startswith("Baris tanpa spasi yang direnggangkan 0,3 em: ")
        assert dict(methods["staged"]["settings"])["Yang berubah"].endswith("fase6_ctrl: argumen data sama, langkah tambahan; "
                                                                             "fase7_track: jarak antar suku kata")
        # Batasan yang dicatat proyek (CLAUDE.md), bila berkas fontnya ada di mesin ini: font latih bercacat, font uji
        # sekeluarga dengan font latih, dan run di rantai yang dilatih sebelum aturan font berspasi sempit.
        if all((ROOT / "fonts/extra" / name).exists() for name in ("BasaJan.ttf", "NewKramawirya.ttf", "CarakanJawa.otf")):
            assert "BasaJan" in methods["render"]["note"] and "NewKramawirya" in methods["render"]["note"]
            assert methods["drop_space"]["note"].startswith("Run fase5_fonts di rantai dilatih sebelum aturan font berspasi sempit ada")
            if (em.TEST_FONT_DIR / "javatext.ttf").exists():
                assert methods["gates"]["note"].startswith("Font uji javatext sekeluarga dengan font latih CarakanJawa (lebar 72 dari 73 ")
                assert methods["probe"]["note"].startswith("Citranya dirender dengan font uji javatext, yang sekeluarga dengan font latih CarakanJawa")

    dump = json.dumps(card, ensure_ascii=False)
    with (ROOT / "data/real/nusaaksara/labels.tsv").open(encoding="utf-8", newline="") as f:
        labels = [line.split("\t")[1] for line in f.read().splitlines()[1:]]
    assert not any(label in dump for label in labels if len(label) >= 4)

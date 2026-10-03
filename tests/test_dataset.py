"""Test dataset & collate — Fase 3. Di sinilah bug CTC paling sering bersembunyi."""

from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

from src.dataset import (
    DOWNSAMPLE,
    H,
    MIN_FRAMES_PER_TARGET,
    TRAIN_FONTS,
    LengthBucketSampler,
    SyntheticLines,
    collate,
    is_trainable,
    to_tensor,
)
from src.model import CRNN
from src.tokenizer import Tokenizer, default_font_paths

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_JSON = ROOT / "data" / "tokenizer.json"
TRAIN_SPLIT = ROOT / "data" / "splits" / "train.txt"

# Keluaran transliterator untuk kalimat Jawa nyata.
SAMPLE_LINES = [
    "ꦧꦱ ꦗꦮ ꦲꦶꦏꦸ ꦧꦱ ꦏꦁ ꦢꦶꦲꦼꦁꦒꦺꦴ ꦢꦼꦤꦶꦁ ꦮꦺꦴꦁ ꦗꦮ꧉",
    "ꦏꦸꦛ ꦪꦺꦴꦒꦾꦏꦂꦠ ꦲꦤ ꦲꦶꦁ ꦠ꧀ꦭꦠꦃ ꦗꦮ ꦠꦼꦔꦃ꧈",
    "ꦮꦺꦴꦁ꧈ ꦏꦺꦴꦮꦺ꧈ ꦚꦮꦶꦗꦶ꧈ ꦏꦿꦠꦺꦴꦤ꧀ ꦥꦿꦝꦤ꧈ ꦱꦱ꧀ꦠꦿ",
    "ꦲꦶꦁ ꦱꦶꦱꦶꦃ ꦭꦺꦴꦂ ꦮꦺꦠꦤ꧀ ꧌ꦏꦶꦢꦸꦭ꧀꧍ ꦲꦤ ꦏꦭꦶ꧇",
]


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    built, _ = Tokenizer.build(SAMPLE_LINES, default_font_paths())
    # Tanpa HarfBuzz, persis seperti di worker DataLoader.
    return Tokenizer(built.charset, built.reorder)


@pytest.fixture(scope="module")
def dataset(tok) -> SyntheticLines:
    return SyntheticLines(SAMPLE_LINES, tok, TRAIN_FONTS, deterministic=True)


def test_collate_in_lens_come_from_original_widths():
    widths = (40, 403, 1000)
    xs = [torch.ones(1, H, w) for w in widths]
    ys = [torch.tensor([1, 2]), torch.tensor([3]), torch.tensor([4, 5, 6])]
    X, Y, in_lens, tgt_lens = collate(list(zip(xs, ys)))
    assert X.shape == (3, 1, H, 1000)
    assert in_lens.tolist() == [w // DOWNSAMPLE for w in widths]  # bukan 1000 // 4 untuk semua
    assert tgt_lens.tolist() == [2, 1, 3]
    assert Y.tolist() == [1, 2, 3, 4, 5, 6]
    assert X[0, :, :, 40:].abs().sum() == 0 and X[1, :, :, 403:].abs().sum() == 0


def test_collate_drops_rejected_samples():
    sample = (torch.ones(1, H, 80), torch.tensor([1]))
    X, _, _, _ = collate([None, sample, None])
    assert X.shape[0] == 1
    assert collate([None, None]) is None


def test_is_trainable_uses_margin():
    assert is_trainable(n_targets=10, img_width=60)
    assert not is_trainable(n_targets=10, img_width=59)


def test_to_tensor_ink_is_one_background_zero():
    img = Image.new("L", (300, 120), 255)
    ImageDraw.Draw(img).rectangle([100, 30, 199, 89], fill=0)
    x = to_tensor(img)
    assert x.shape == (1, H, 240)
    assert x[0, 0, 0] == 0
    assert x[0, H // 2, 120] == 1


def test_to_tensor_normalizes_gray_paper_and_faded_ink():
    img = Image.new("L", (300, 120), 190)
    ImageDraw.Draw(img).rectangle([100, 30, 199, 89], fill=90)
    x = to_tensor(img)
    assert abs(float(x[0, 0, 0])) < 1e-6
    assert abs(float(x[0, H // 2, 120]) - 1.0) < 1e-6


def test_to_tensor_does_not_amplify_blank_image():
    img = Image.new("L", (200, 100), 250)
    ImageDraw.Draw(img).rectangle([10, 10, 20, 20], fill=245)
    assert float(to_tensor(img).max()) < 0.05


def test_item_shape_and_visual_target(dataset, tok):
    for i, text in enumerate(SAMPLE_LINES):
        x, y = dataset[i]
        assert x.shape[:2] == (1, H)
        assert y.tolist() == tok.encode(tok.to_visual(text))
        assert int(y.min()) >= 1


def test_deterministic_mode_repeats(dataset):
    assert torch.equal(dataset[1][0], dataset[1][0])


def test_no_sample_in_batch_violates_min_frames(dataset):
    X, Y, in_lens, tgt_lens = collate([dataset[i] for i in range(len(dataset))])
    assert (in_lens.float() >= MIN_FRAMES_PER_TARGET * tgt_lens.float()).all()


def test_rendering_never_writes_images(dataset, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("citra training disimpan ke disk")

    monkeypatch.setattr(Image.Image, "save", refuse)
    for i in range(len(dataset)):
        dataset[i]


def test_ctc_loss_is_finite_on_rendered_batch(dataset, tok):
    X, Y, in_lens, tgt_lens = collate([dataset[i] for i in range(len(dataset))])
    model = CRNN(tok.n_classes)
    logits = model(X, in_lens)
    assert logits.shape[1] == X.shape[-1] // DOWNSAMPLE
    log_probs = logits.log_softmax(-1).permute(1, 0, 2)
    loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)(log_probs, Y, in_lens, tgt_lens)
    assert torch.isfinite(loss)
    loss.backward()


def test_length_bucket_sampler_covers_every_index_once():
    lines = ["x" * n for n in range(1, 101)]
    sampler = LengthBucketSampler(lines, batch_size=8)
    seen = sorted(i for batch in sampler for i in batch)
    assert seen == list(range(100))


@pytest.mark.skipif(not (TOKENIZER_JSON.exists() and TRAIN_SPLIT.exists()), reason="korpus belum dibangun")
def test_corpus_lines_are_trainable():
    tokenizer = Tokenizer.load(TOKENIZER_JSON)
    lines = TRAIN_SPLIT.read_text(encoding="utf-8").splitlines()[:300]
    ds = SyntheticLines(lines, tokenizer, TRAIN_FONTS, deterministic=True)
    items = [ds[i] for i in range(len(ds))]
    rejected = sum(item is None for item in items)
    print(f"\nditolak filter T>=1.5L: {rejected}/{len(items)}")
    X, Y, in_lens, tgt_lens = collate(items)
    assert (in_lens.float() >= MIN_FRAMES_PER_TARGET * tgt_lens.float()).all()
    assert rejected / len(items) < 0.01


def test_long_pasangan_chain_renders_without_clipping():
    # Baris korpus yang menghentikan training 4b: rantai pasangan dari kata serapan.
    from src.dataset import RENDER_SIZE_RANGE
    from src.render import render_line

    text = "ꦲꦲꦸꦱ꧀ꦠ꧀ꦧ꧀ꦪ꧀ꦒ꧀ꦢ ꦥꦸꦤꦶꦏ ꦝꦸꦱꦸꦤ꧀ ꦲꦶꦁꦏꦁ ꦏꦥꦼꦂꦤꦃ ꦲꦶꦁ"
    for font in TRAIN_FONTS:
        for size in RENDER_SIZE_RANGE:
            img = render_line(text, font, size)
            assert img.width > 0 and img.height > 0


def test_unrenderable_sample_is_skipped(tok, monkeypatch):
    from src.render import RenderClipped

    def clipped(*args, **kwargs):
        raise RenderClipped("uji")

    ds = SyntheticLines(SAMPLE_LINES, tok, TRAIN_FONTS, deterministic=True)
    monkeypatch.setattr("src.dataset.render_line", clipped)
    with pytest.warns(UserWarning, match="dilewati"):
        assert ds[0] is None


def test_missing_raqm_is_not_swallowed(tok, monkeypatch):
    def no_raqm(*args, **kwargs):
        raise RuntimeError("RAQM tidak aktif")

    ds = SyntheticLines(SAMPLE_LINES, tok, TRAIN_FONTS, deterministic=True)
    monkeypatch.setattr("src.dataset.render_line", no_raqm)
    with pytest.raises(RuntimeError, match="RAQM"):
        ds[0]


def test_drop_space_renders_and_labels_the_same_text(tok):
    ds = SyntheticLines(SAMPLE_LINES, tok, TRAIN_FONTS, deterministic=True, drop_space_prob=1.0)
    for i, text in enumerate(SAMPLE_LINES):
        _, used = ds.sample(i)
        assert used == text.replace(" ", "")
        _, y = ds[i]
        assert y.tolist() == tok.encode(tok.to_visual(used))


def test_drop_space_zero_leaves_deterministic_renders_unchanged(tok):
    # Probabilitas 0 tidak boleh mengonsumsi angka acak: render val lama tetap identik.
    plain = SyntheticLines(SAMPLE_LINES, tok, TRAIN_FONTS, deterministic=True, seed=7)
    explicit = SyntheticLines(SAMPLE_LINES, tok, TRAIN_FONTS, deterministic=True, seed=7, drop_space_prob=0.0)
    assert torch.equal(plain[2][0], explicit[2][0])


def test_space_ratio_flags_only_narrow_space_fonts():
    from src.dataset import MIN_SPACE_RATIO, ROOT, space_ratio

    for font in TRAIN_FONTS:
        assert space_ratio(str(font)) >= MIN_SPACE_RATIO, font.name
    narrow = ROOT / "fonts" / "extra" / "GumregahNew.ttf"
    if narrow.exists():
        assert space_ratio(str(narrow)) < MIN_SPACE_RATIO


def test_narrow_space_font_drops_spaces_from_render_and_label(tok, monkeypatch):
    monkeypatch.setattr("src.dataset.space_ratio", lambda font: 0.0)
    ds = SyntheticLines(SAMPLE_LINES, tok, TRAIN_FONTS, deterministic=True)
    assert any(" " in text for text in SAMPLE_LINES)
    for i, text in enumerate(SAMPLE_LINES):
        _, used = ds.sample(i)
        assert used == text.replace(" ", "")
        assert ds[i][1].tolist() == tok.encode(tok.to_visual(used))


def test_normal_space_font_keeps_spaces(tok):
    ds = SyntheticLines(SAMPLE_LINES, tok, TRAIN_FONTS, deterministic=True)
    assert [ds.sample(i)[1] for i in range(len(SAMPLE_LINES))] == list(SAMPLE_LINES)


@pytest.mark.skipif(not TOKENIZER_JSON.exists(), reason="tokenizer belum dibangun")
def test_rare_text_prefixes_opener_and_inserts_rare_codepoint():
    from src.text_augment import LINE_OPENERS, RareText

    # Tokenizer asli: tokenizer kecil fixture `tok` tidak memuat aksara langka maupun adeg-adeg.
    tok = Tokenizer.load(TOKENIZER_JSON)
    rare = RareText.from_lines(SAMPLE_LINES * 20, tok.charset, insert_prob=1.0, opener_prob=1.0)
    ds = SyntheticLines(SAMPLE_LINES, tok, TRAIN_FONTS, deterministic=True, rare_text=rare)
    with_rare = 0
    for i in range(len(SAMPLE_LINES)):
        _, used = ds.sample(i)
        assert used.startswith(LINE_OPENERS[0])
        with_rare += any(ch in rare.rare for ch in used[1:])
        x, y = ds[i]
        assert y.tolist() == tok.encode(tok.to_visual(used))
    assert with_rare >= len(SAMPLE_LINES) - 1


def test_inactive_rare_text_leaves_deterministic_renders_unchanged(tok):
    from src.text_augment import RareText

    plain = SyntheticLines(SAMPLE_LINES, tok, TRAIN_FONTS, deterministic=True, seed=3, drop_space_prob=0.5)
    inactive = SyntheticLines(SAMPLE_LINES, tok, TRAIN_FONTS, deterministic=True, seed=3, drop_space_prob=0.5,
                              rare_text=RareText.from_lines(SAMPLE_LINES, tok.charset))
    for i in range(len(SAMPLE_LINES)):
        assert plain.sample(i)[1] == inactive.sample(i)[1]
        assert torch.equal(plain[i][0], inactive[i][0])

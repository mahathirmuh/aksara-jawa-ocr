"""Fase 3 — dataset render on-the-fly dan collate untuk CTC.

Citra tidak pernah disimpan ke disk: setiap sampel dirender ulang di worker
DataLoader dari teks korpus dan font, lalu diskalakan ke tinggi H.
"""

import random
import warnings
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from src.render import EmptyRender, RenderClipped, load_font, render_line
from src.spelling import PROBES, standardize
from src.tokenizer import Tokenizer, is_well_formed

if TYPE_CHECKING:
    from src.text_augment import RareText

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FONTS = sorted(p for p in (ROOT / "fonts").glob("*") if p.suffix.lower() in {".ttf", ".otf"})

H = 96
# Total stride lebar CNN. HARUS sama dengan stride model sebenarnya;
# tests/test_shapes.py memverifikasinya pada output model.
DOWNSAMPLE = 4
MIN_FRAMES_PER_TARGET = 1.5
RENDER_SIZE_RANGE = (56, 72)
# Celah kata (x lebar ka) di bawah ini nyaris tak tampak, jadi spasi di label tidak punya
# padanan di citra. Terukur: GumregahNew 0,09, abmAksaJawa 0,10; ARDemak 0,14, Tuladha OT 0,18,
# Noto & javatext 0,21.
MIN_SPACE_RATIO = 0.12

Augment = Callable[[Image.Image, random.Random], Image.Image]


MIN_CONTRAST = 16  # selisih latar-tinta (0-255) di bawah ini dianggap citra polos


def to_tensor(img: Image.Image) -> torch.Tensor:
    """Citra teks gelap di atas latar terang -> tensor (1, H, W) dengan tinta = 1.

    Tinta = 1 dan latar = 0 supaya padding nol di collate sama dengan latar.
    Kontras dinormalisasi per citra: latar (median, baris teks didominasi latar)
    -> 0 dan tinta (persentil 1) -> 1. Model dilatih pada latar persis 0; tanpa ini
    kertas abu-abu saja menaikkan CER dari 0,9% ke 74,5%. Render bersih tidak berubah.
    """
    img = img.convert("L")
    width = max(DOWNSAMPLE, round(img.width * H / img.height))
    img = img.resize((width, H), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    background = float(np.percentile(arr, 50))
    ink = float(np.percentile(arr, 1))
    if background - ink < MIN_CONTRAST:
        x = 1.0 - arr / 255.0  # citra polos: jangan perbesar derau
    else:
        x = np.clip((background - arr) / (background - ink), 0.0, 1.0)
    return torch.from_numpy(x.astype(np.float32)).unsqueeze(0)


def is_trainable(n_targets: int, img_width: int, downsample: int = DOWNSAMPLE) -> bool:
    """CTC butuh T >= L; dengan margin 1.5x supaya repetisi + blank muat."""
    return (img_width // downsample) >= n_targets * MIN_FRAMES_PER_TARGET


@lru_cache(maxsize=None)
def space_ratio(font_path: str) -> float:
    """Lebar celah kata hasil shaping dibagi lebar ka (GumregahNew: advance spasi 0, celah 0,09)."""
    font = load_font(font_path, 64)
    ka = "ꦏ"
    return (font.getlength(f"{ka}{ka} {ka}{ka}") - font.getlength(ka * 4)) / font.getlength(ka)


# Pasangan yang tidak menumpuk: lebar "ka + pangkon + ka" dibagi lebar "ka". Terukur 2026-10-16 di lingkungan training:
# font latih lain 1,00-1,39; BasaJan 2,56 di .venv-xpu (GSUB tidak diterapkan Pillow 12.3 / HarfBuzz 14.2), 1,00 di .venv.
MAX_PASANGAN_WIDTH = 1.8
# Urutan yang sebuah font gambar SAMA dengan urutan lain (dua label untuk satu gambar): urutan -> pembandingnya.
# NewKramawirya menggambar pada lungsi sesudah pangkon dengan glyph pada lingsa (substitusi kontekstual).
RENDER_CONFLICTS = {"꧀꧉": "꧀꧈"}


def oblique(img: Image.Image, shear: float) -> Image.Image:
    """Citra dimiringkan ke kanan seperti huruf italic: bagian atas bergeser ke kanan sejauh `shear` x tinggi. Kanvas
    dilebarkan supaya tidak ada tinta yang terpotong; latar putih seperti render."""
    w, h = img.size
    return img.transform((w + int(round(shear * h)) + 1, h), Image.AFFINE, (1, shear, -shear * h, 0, 1, 0),
                         resample=Image.Resampling.BILINEAR, fillcolor=255)


@lru_cache(maxsize=None)
def shaping_ok(font_path: str) -> bool:
    """Pasangan font ini menumpuk di interpreter yang sedang berjalan (shaping berbeda antar lingkungan)."""
    def width(text: str) -> int:
        cols = np.nonzero((np.asarray(render_line(text, font_path, 64).convert("L")) < 128).any(axis=0))[0]
        return int(cols[-1] - cols[0] + 1)

    return width("ꦏ꧀ꦏ") / width("ꦏ") < MAX_PASANGAN_WIDTH


@lru_cache(maxsize=None)
def render_conflicts(font_path: str) -> frozenset[str]:
    """Urutan RENDER_CONFLICTS yang di font ini tergambar sama persis dengan pembandingnya."""
    def same(a: str, b: str) -> bool:
        x, y = (np.asarray(render_line("ꦏ" + text, font_path, 64).convert("L")) < 128 for text in (a, b))
        return x.shape == y.shape and not (x ^ y).any()

    return frozenset(seq for seq, other in RENDER_CONFLICTS.items() if same(seq, other))


@lru_cache(maxsize=None)
def spelling_twins(font_path: str) -> frozenset[str]:
    """Aturan ejaan baku yang di font ini TIDAK mengubah citra: ejaan korpus dan ejaan baku dirender kembar (GumregahNew
    meligaturkan cakra + pepet menjadi keret). Diperiksa dengan interpreter yang sedang berjalan, karena shaping berbeda
    antar lingkungan."""
    def same(a: str, b: str) -> bool:
        x, y = (np.asarray(render_line(text, font_path, 64).convert("L")) < 128 for text in (a, b))
        return x.shape == y.shape and not (x ^ y).any()

    return frozenset(name for name, (before, after) in PROBES.items() if same(before, after))


class SyntheticLines(Dataset):
    """Baris korpus dirender on-the-fly -> (citra (1, H, W), target urutan visual).

    `deterministic=True` memberi citra yang sama untuk indeks yang sama
    (validasi/test). Sampel yang melanggar T >= 1.5L dikembalikan sebagai None
    dan dibuang oleh collate.
    """

    def __init__(
        self,
        lines: Sequence[str],
        tokenizer: Tokenizer,
        font_paths: Sequence[str | Path] = TRAIN_FONTS,
        size_range: tuple[int, int] = RENDER_SIZE_RANGE,
        augment: Augment | None = None,
        deterministic: bool = False,
        seed: int = 0,
        drop_space_prob: float = 0.0,
        rare_text: "RareText | None" = None,
        track_prob: float = 0.0,
        track_max: float = 0.0,
        standard_prob: float = 0.0,
        slant_prob: float = 0.0,
        slant_max: float = 0.0,
        avoid_conflicts: bool = False,
        neighbor_prob: float = 0.0,
    ):
        if not font_paths:
            raise ValueError("Butuh minimal satu font")
        self.neighbor_prob = neighbor_prob
        self.slant_prob = slant_prob
        self.slant_max = slant_max
        self.avoid_conflicts = avoid_conflicts
        self.rare_text = rare_text
        self.track_prob = track_prob
        self.track_max = track_max
        self.standard_prob = standard_prob
        self.lines = list(lines)
        self.tokenizer = tokenizer
        self.font_paths = [str(p) for p in font_paths]
        self.size_range = size_range
        self.augment = augment
        self.deterministic = deterministic
        self.seed = seed
        self.drop_space_prob = drop_space_prob

    def __len__(self) -> int:
        return len(self.lines)

    def _rng(self, idx: int) -> random.Random:
        # Non-deterministik: random.Random() diseed dari os.urandom, jadi setiap
        # worker DataLoader mendapat aliran acak sendiri.
        return random.Random(f"{self.seed}:{idx}") if self.deterministic else random.Random()

    def _round_trips(self, text: str) -> bool:
        """Semua karakter ada di charset dan urutan visual kembali ke teks yang sama."""
        charset = set(self.tokenizer.charset)
        return all(ch in charset for ch in text) and self.tokenizer.to_logical(self.tokenizer.to_visual(text)) == text

    def sample(self, idx: int) -> tuple[Image.Image, str]:
        """Citra dan teks yang benar-benar dirender untuk indeks ini."""
        rng = self._rng(idx)
        text = self.lines[idx]
        # getattr: dataset yang di-pickle oleh versi sebelumnya tidak punya atribut ini.
        drop_space = getattr(self, "drop_space_prob", 0.0)
        # Urutan angka acak (drop, font, ukuran) sama dengan versi sebelumnya: render
        # val/test deterministik tidak berubah.
        drop = bool(drop_space) and rng.random() < drop_space
        font = rng.choice(self.font_paths)
        # Font yang menggambar sebuah urutan di baris ini sama dengan urutan lain diganti (undian tambahan hanya bila itu
        # terjadi, jadi baris lain dan render tanpa pilihan ini tidak berubah).
        if getattr(self, "avoid_conflicts", False):
            joined = text.replace(" ", "")
            if any(seq in joined for seq in render_conflicts(font)):
                others = [f for f in self.font_paths if not any(seq in joined for seq in render_conflicts(f))]
                if others:
                    font = rng.choice(others)
        size = rng.randint(*self.size_range)
        # Ejaan baku (src/spelling.py) pada sebagian baris, sebelum sisipan dan sebelum spasi dibuang (aturan nya di
        # depan ca/ja berlaku di dalam kata saja). Probabilitas 0 tidak mengonsumsi angka acak, jadi render val/test
        # dan run lama tidak berubah. Aturan yang glyph-nya tidak ada di font terpilih dilewati.
        # Font yang merender kedua ejaan KEMBAR untuk sebuah aturan selalu memakai ejaan baku aturan itu, supaya satu
        # gambar tidak berlabel dua (seperti spasi pada font berspasi sempit).
        standard_prob = getattr(self, "standard_prob", 0.0)
        if standard_prob:
            chosen = rng.random() < standard_prob
            twins = spelling_twins(font)
            if chosen or twins:
                from src.text_augment import font_codepoints  # impor di sini: text_augment memuat uharfbuzz

                text = standardize(text, font_codepoints(font), lambda new: is_well_formed(new) and self._round_trips(new),
                                   None if chosen else twins)
        rare = getattr(self, "rare_text", None)
        if rare is not None and rare.active:
            # Sebelum spasi dibuang: angka & pada disisipkan di batas kata. Probabilitas 0 tidak
            # mengonsumsi angka acak, jadi render val (tanpa rare_text) tidak berubah.
            text = rare(text, rng, font, self._round_trips)
        if drop or (" " in text and space_ratio(font) < MIN_SPACE_RATIO):
            # Buku cetak dan papan nama Jawa umumnya ditulis tanpa spasi. Menghapus
            # spasi di Unicode membuat "C + pangkon + C" antar kata dirender sebagai
            # pasangan, persis ortografi aslinya; label ikut teks yang sama. Font yang
            # spasinya nyaris tak tampak selalu begini, supaya label tidak memuat spasi
            # yang tidak ada di citra.
            text = text.replace(" ", "")
        # Jarak antar suku kata acak per baris (em). Diundi paling akhir dan hanya bila aktif, jadi
        # render val/test deterministik (tanpa tracking) tidak berubah.
        tracking = 0.0
        track_prob = getattr(self, "track_prob", 0.0)
        if track_prob and rng.random() < track_prob:
            tracking = rng.uniform(0.0, self.track_max)
        img = render_line(text, font, size, tracking=tracking) if tracking else render_line(text, font, size)
        # Kemiringan seperti huruf italic cetakan, diundi sesudah semua undian lain dan hanya bila aktif (render val/test
        # dan run lama tidak berubah). Geometri, jadi sebelum degradasi kertas/tinta/optik.
        slant_prob = getattr(self, "slant_prob", 0.0)
        if slant_prob and rng.random() < slant_prob:
            img = oblique(img, rng.uniform(0.0, self.slant_max))
        # Pecahan baris tetangga di tepi atas dan bawah (src/neighbors.py), diundi sesudah semua undian lain dan hanya
        # bila aktif (render val/test dan run lama tidak berubah). Geometri halaman, jadi sebelum degradasi.
        neighbor_prob = getattr(self, "neighbor_prob", 0.0)
        if neighbor_prob and rng.random() < neighbor_prob:
            img = self._with_neighbors(img, font, size, rng)
        return (self.augment(img, rng) if self.augment else img), text

    def _with_neighbors(self, img: Image.Image, font: str, size: int, rng: random.Random) -> Image.Image:
        """Pita bawah sebuah baris lain di atas citra dan pita atas baris lain di bawahnya (src/neighbors.py); tiap
        sisi kosong dengan peluang 0,3. Baris tetangga diambil acak dari kumpulan baris yang sama dan dirender dengan font
        dan ukuran yang sama, tanpa spasi (cetakan umumnya tanpa spasi). Label tetap teks baris ini."""
        from src.neighbors import with_neighbors  # impor di sini: hanya dipakai bila opsi ini hidup

        def neighbor() -> Image.Image | None:
            if rng.random() < 0.3:
                return None
            other = self.lines[rng.randrange(len(self.lines))].replace(" ", "")
            try:
                return render_line(other, font, size)
            except (RenderClipped, EmptyRender):
                return None

        return with_neighbors(img, neighbor(), neighbor(), rng)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        try:
            img, text = self.sample(idx)
        except (RenderClipped, EmptyRender) as e:
            # Satu baris yang tidak bisa dirender tidak boleh menghentikan training
            # berjam-jam; collate membuangnya. Kegagalan lain (mis. RAQM mati) tetap
            # dilempar: menelannya akan membuang SEMUA sampel secara senyap.
            warnings.warn(f"Sampel {idx} dilewati: {e}")
            return None
        x = to_tensor(img)
        y = torch.tensor(self.tokenizer.encode(self.tokenizer.to_visual(text)), dtype=torch.long)
        if not is_trainable(len(y), x.shape[-1]):
            return None
        return x, y


def collate(batch):
    """Pad ke lebar terbesar. in_lens dari lebar ASLI tiap citra, BUKAN lebar padded."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys = zip(*batch)
    widths = [x.shape[-1] for x in xs]
    X = torch.zeros(len(xs), 1, H, max(widths))
    for i, x in enumerate(xs):
        X[i, :, :, : x.shape[-1]] = x
    in_lens = torch.tensor([max(1, w // DOWNSAMPLE) for w in widths], dtype=torch.long)
    tgt_lens = torch.tensor([len(y) for y in ys], dtype=torch.long)
    Y = torch.cat(ys)  # CTC menerima target 1D terkonkatenasi
    return X, Y, in_lens, tgt_lens


class LengthBucketSampler(Sampler[list[int]]):
    """Batch berisi baris dengan panjang teks mirip, supaya padding (dan komputasi) minim."""

    def __init__(self, lines: Sequence[str], batch_size: int, shuffle: bool = True, seed: int = 0):
        self.lengths = [len(line) for line in lines]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        order = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(order)
        # Urutkan per kelompok besar agar tetap acak antar-epoch.
        chunk = self.batch_size * 50
        batches = []
        for start in range(0, len(order), chunk):
            group = sorted(order[start : start + chunk], key=self.lengths.__getitem__)
            batches += [group[i : i + self.batch_size] for i in range(0, len(group), self.batch_size)]
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

// Transliterasi Latin -> aksara Jawa per kata untuk korpus Fase 2.
//
// Pemakaian: node translit.js <kata.txt> <hasil.jsonl>
// Masukan satu kata per baris (huruf kecil); keluaran satu JSON {aks, back}
// per baris dengan urutan yang sama. `back` = aks ditransliterasi balik ke
// Latin, dipakai src/corpus.py untuk filter round-trip.
"use strict";

const fs = require("fs");
const { toHonocoroko, fromHonocoroko } = require("@naandalist/honocoroko");

const PANGKON = "꧀";
// useSwara:false -> vokal awal ditulis dengan ha (ꦲꦤ untuk "ana"), sesuai
// ortografi baku; aksara swara disisakan untuk injeksi codepoint langka.
const OPTIONS = { useSwara: false, useMurda: false };

// honocoroko tidak memberi pangkon pada konsonan penutup kata: "taun" -> ꦠꦈꦤ,
// yang terbaca "tauna". Akhiran -ng/-r/-h sudah menjadi cecak/layar/wignyan.
function needsFinalPangkon(core) {
  if (!/[a-z]$/.test(core) || /[aiueoé]$/.test(core)) return false;
  return !/(ng|r|h)$/.test(core);
}

// honocoroko hanya membentuk aksara digraf bila diikuti "a": "thi" -> ta+pangkon+ha,
// "ngi" -> na+pangkon+ga, "nyu" -> na+pengkal. README-nya sendiri menyatakan nya/nga
// harus ꦚ/ꦔ (bukan n+pengkal / cecak), jadi bentuk terpecah dikembalikan ke digrafnya.
// Filter round-trip tidak menangkap ini karena arah balik membacanya sebagai digraf juga.
const DIGRAPH_FIXES = [
  [/ꦠ꧀ꦲ/g, "ꦛ"], // th -> ꦛ
  [/ꦢ꧀ꦲ/g, "ꦝ"], // dh -> ꦝ
  [/ꦤ꧀ꦒ/g, "ꦔ"], // ng -> ꦔ
  [/ꦤꦾ/g, "ꦚ"], // ny -> ꦚ
];

function fixDigraphs(aks) {
  return DIGRAPH_FIXES.reduce((s, [pattern, replacement]) => s.replace(pattern, replacement), aks);
}

function transliterate(word) {
  const [, lead, core, trail] = word.match(/^([(]*)(.*?)([,.:)]*)$/);
  let aks = toHonocoroko(lead) + fixDigraphs(toHonocoroko(core, OPTIONS));
  if (needsFinalPangkon(core)) aks += PANGKON;
  aks += toHonocoroko(trail);
  return { aks, back: fromHonocoroko(aks) };
}

const [inPath, outPath] = process.argv.slice(2);
const words = fs.readFileSync(inPath, "utf8").split(/\r?\n/);
if (words.length && words[words.length - 1] === "") words.pop();
fs.writeFileSync(outPath, words.map((w) => JSON.stringify(transliterate(w))).join("\n") + "\n");

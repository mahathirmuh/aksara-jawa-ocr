<?php

namespace Tests\Unit;

use App\Support\AksaraWriter;
use App\Support\Transliterator;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;

/** Latin -> aksara Jawa menurut aturan. Ejaan yang diharapkan sama dengan ejaan lema itu di kamus (Wiktionary). */
class AksaraWriterTest extends TestCase
{
    public static function samples(): array
    {
        return [
            'vokal awal memakai ha, penutup kata berpangkon' => ['aku mangan sega', 'ꦲꦏꦸ ꦩꦔꦤ꧀ ꦱꦼꦒ'],
            'pasangan di tengah kata' => ['bakso', 'ꦧꦏ꧀ꦱꦺꦴ'],
            'vokal berurutan' => ['taun', 'ꦠꦲꦸꦤ꧀'],
            'cecak' => ['bangsa', 'ꦧꦁꦱ'],
            'layar' => ['pinter', 'ꦥꦶꦤ꧀ꦠꦼꦂ'],
            'wignyan dan taling-tarung' => ['omah', 'ꦲꦺꦴꦩꦃ'],
            'dh' => ['dhahar', 'ꦝꦲꦂ'],
            'th' => ['thithik', 'ꦛꦶꦛꦶꦏ꧀'],
            'ny' => ['nyata', 'ꦚꦠ'],
            'ng pembuka + taling' => ['ngombé', 'ꦔꦺꦴꦩ꧀ꦧꦺ'],
            'ng + konsonan di awal kata' => ['nglakoni', 'ꦔ꧀ꦭꦏꦺꦴꦤꦶ'],
            'ng + cakra' => ['ngrungu', 'ꦔꦿꦸꦔꦸ'],
            'cakra' => ['putra', 'ꦥꦸꦠꦿ'],
            'pengkal' => ['setya', 'ꦱꦼꦠꦾ'],
            'keret' => ['tresna', 'ꦠꦽꦱ꧀ꦤ'],
            'pa cerek' => ['arep', 'ꦲꦉꦥ꧀'],
            'pa cerek + cecak' => ['bareng', 'ꦧꦉꦁ'],
            'nga lelet' => ['lemah', 'ꦊꦩꦃ'],
            'la berpepet sebagai pasangan' => ['klepon', 'ꦏ꧀ꦭꦼꦥꦺꦴꦤ꧀'],
            'taling é' => ['gedhé', 'ꦒꦼꦝꦺ'],
            'taling è' => ['kabèh', 'ꦏꦧꦺꦃ'],
            'taling tanpa konsonan pembuka' => ['saé élok', 'ꦱꦲꦺ ꦲꦺꦭꦺꦴꦏ꧀'],
            'n di depan c ditulis nya' => ['panci', 'ꦥꦚ꧀ꦕꦶ'],
            'n di depan j ditulis nya' => ['panjenengan', 'ꦥꦚ꧀ꦗꦼꦤꦼꦔꦤ꧀'],
            'layar sebelum ya' => ['karya', 'ꦏꦂꦪ'],
            'aksara rekan' => ['foto zaman', 'ꦥ꦳ꦺꦴꦠꦺꦴ ꦗ꦳ꦩꦤ꧀'],
            'kata ulang tanpa tanda hubung' => ['alon-alon', 'ꦲꦭꦺꦴꦤ꧀ꦲꦭꦺꦴꦤ꧀'],
            'angka diapit pada pangkat, koma dan titik' => ['taun 1945, ing Jakarta.', 'ꦠꦲꦸꦤ꧀ ꧇꧑꧙꧔꧕꧇꧈ ꦲꦶꦁ ꦗꦏꦂꦠ꧉'],
            'jam dan titik dua sesudah angka' => ['jam 10:30, taun 1945: perang', 'ꦗꦩ꧀ ꧇꧑꧐:꧓꧐꧇꧈ ꦠꦲꦸꦤ꧀ ꧇꧑꧙꧔꧕꧇ ꦥꦼꦫꦁ'],
            'huruf besar dan pepet bertanda' => ['Gêdhé', 'ꦒꦼꦝꦺ'],
            'transliterasi ilmiah' => ['banḍa', 'ꦧꦤ꧀ꦝ'],
            'apostrof dibuang, tanda tanya dibiarkan' => ["Jum'at, apa?", 'ꦗꦸꦩꦠ꧀꧈ ꦲꦥ?'],
            'tanda hubung berspasi dibiarkan' => ['a - b', 'ꦲ - ꦧ꧀'],
        ];
    }

    #[DataProvider('samples')]
    public function test_writes_javanese_script(string $latin, string $aksara): void
    {
        $this->assertSame($aksara, AksaraWriter::fromLatin($latin));
    }

    public function test_corpus_convention_skips_the_standard_spelling_rules(): void
    {
        // Konvensi korpus OCR (alat lama): tanpa cerek, lelet, keret, nya di depan c/j, dan pada pangkat.
        $this->assertSame('ꦭꦼꦩꦃ ꦫꦼꦒ ꦠꦿꦼꦱ꧀ꦤ ꦥꦤ꧀ꦕꦶ ꧒꧐꧒꧖', AksaraWriter::fromLatin('lemah rega tresna panci 2026', false));
        $this->assertSame('ꦊꦩꦃ ꦉꦒ ꦠꦽꦱ꧀ꦤ ꦥꦚ꧀ꦕꦶ ꧇꧒꧐꧒꧖꧇', AksaraWriter::fromLatin('lemah rega tresna panci 2026'));
    }

    public function test_names_use_the_dictionary_spelling_first_then_the_murda_and_swara_rule(): void
    {
        // Bawaannya mati: huruf besar tidak mengubah apa pun, jadi hasil lama dan angka kecocokan kamus tetap.
        $text = 'Aku lunga menyang Surabaya karo Bima.';
        $this->assertSame('ꦲꦏꦸ ꦭꦸꦔ ꦩꦼꦚꦁ ꦱꦸꦫꦧꦪ ꦏꦫꦺꦴ ꦧꦶꦩ꧉', AksaraWriter::fromLatin($text));
        $this->assertSame(AksaraWriter::fromLatin(mb_strtolower($text)), AksaraWriter::fromLatin($text));

        // Lapis 1: nama yang ada di kamus memakai ejaan kamus (Bima: ba murda). Lapis 2: nama lain mendapat satu
        // murda pada konsonan pertama yang punya (Surabaya: sa). Kata pertama kalimat bukan nama (Aku).
        $this->assertSame(['aksara' => 'ꦲꦏꦸ ꦭꦸꦔ ꦩꦼꦚꦁ ꦯꦸꦫꦧꦪ ꦏꦫꦺꦴ ꦨꦶꦩ꧉', 'dictionary' => ['bima'], 'rule' => ['surabaya']],
            AksaraWriter::write($text));
        $this->assertSame(AksaraWriter::write($text)['aksara'], AksaraWriter::fromLatin($text, names: true));
        // Ejaan kamus berlaku juga di awal kalimat, dan tidak diubah kaidah (Suwarnabhumi: murda di tengah, dua).
        $this->assertSame(['aksara' => 'ꦄꦂꦗꦸꦤ ꦭꦤ꧀ ꦆꦤ꧀ꦢꦿ꧉ ꦱꦸꦮꦂꦟꦨꦹꦩꦶ', 'dictionary' => ['arjuna', 'indra', 'suwarnabhumi'], 'rule' => []],
            AksaraWriter::write('Arjuna lan Indra. Suwarnabhumi'));
        $names = AksaraWriter::dictionaryNames();
        $this->assertSame('ꦨꦶꦩ', $names['bima']);
        $this->assertGreaterThanOrEqual(10, count($names));
        // Nama kamus yang juga kata biasa berhuruf kecil di kamus yang sama (agung = besar): di awal kalimat huruf
        // besarnya bukan tanda nama, jadi ditulis biasa dan tidak dilaporkan; di tengah kalimat tetap ejaan kamus.
        $ordinary = AksaraWriter::ordinaryNames();
        $this->assertArrayHasKey('agung', $ordinary);
        $this->assertSame(['aditya', 'agung', 'arga', 'arya', 'bima', 'ukir'], array_keys($ordinary));  // bima = menakutkan (sastra)
        $this->assertSame(['aksara' => AksaraWriter::fromLatin('bima lunga'), 'dictionary' => [], 'rule' => []], AksaraWriter::write('Bima lunga'));
        $this->assertSame([], array_diff_key($ordinary, $names));
        $this->assertSame(['aksara' => AksaraWriter::fromLatin('agung banget omahe.'), 'dictionary' => [], 'rule' => []],
            AksaraWriter::write('Agung banget omahe.'));
        $this->assertSame(['aksara' => AksaraWriter::fromLatin('omahe').' '.$names['agung'], 'dictionary' => ['agung'], 'rule' => []], AksaraWriter::write('omahe Agung'));
        $this->assertNotSame($names['agung'], AksaraWriter::fromLatin('agung'));
        $this->assertSame([], AksaraWriter::write('Teka. Agung banget')['dictionary']);

        // Kaidah: vokal pembuka -> swara dan sesudah itu tanpa murda (Ésti, Anto); pepet tidak punya swara, jadi
        // murdanya tetap dicari (Eko); n di depan j = nya murda (Njoto); hanya SATU murda per nama (Kartini);
        // penutup suku kata tidak dijadikan murda (Rangga: ga, bukan cecak); nama tanpa keduanya tertulis biasa dan
        // tidak dilaporkan (Rara); tiap bagian nama bertanda hubung dinilai sendiri.
        $write = fn (string $name) => AksaraWriter::write('aku lan '.$name);
        $this->assertSame('ꦲꦏꦸ ꦭꦤ꧀ ꦌꦱ꧀ꦠꦶ', $write('Ésti')['aksara']);
        $this->assertSame('ꦲꦏꦸ ꦭꦤ꧀ ꦄꦤ꧀ꦠꦺꦴ', $write('Anto')['aksara']);
        $this->assertSame('ꦲꦏꦸ ꦭꦤ꧀ ꦲꦼꦑꦺꦴ', $write('Eko')['aksara']);
        $this->assertSame('ꦲꦏꦸ ꦭꦤ꧀ ꦘ꧀ꦗꦺꦴꦠꦺꦴ', $write('Njoto')['aksara']);
        $this->assertSame('ꦲꦏꦸ ꦭꦤ꧀ ꦑꦂꦠꦶꦤꦶ', $write('Kartini')['aksara']);
        $this->assertSame('ꦲꦏꦸ ꦭꦤ꧀ ꦫꦁꦓ', $write('Rangga')['aksara']);
        $this->assertSame(['aksara' => 'ꦲꦏꦸ ꦭꦤ꧀ ꦫꦫꦯꦤ꧀ꦠꦁ', 'dictionary' => [], 'rule' => ['santang']], $write('Rara-Santang'));
        $this->assertSame([], $write('Rara')['rule']);
        // Konsonan penutup suku kata yang diberi pangkon juga tidak dijadikan murda: Thomas dan Marwan tertulis biasa
        // (s dan n menutup suku kata), Yanti mendapat murda pada ta yang membuka suku kata kedua.
        foreach (['Thomas', 'Marwan'] as $name) {
            $this->assertSame(['aksara' => AksaraWriter::fromLatin('aku lan '.mb_strtolower($name)), 'dictionary' => [], 'rule' => []], $write($name));
        }
        $this->assertSame('ꦲꦏꦸ ꦭꦤ꧀ ꦪꦤ꧀ꦡꦶ', $write('Yanti')['aksara']);
        // è pembuka nama ditulis dengan aksara swara yang sama dengan é.
        $this->assertSame($write('Ésti')['aksara'], $write('Èsti')['aksara']);
        // Bilangan di awal kalimat sudah "kata pertama": nama sesudahnya tetap nama.
        $this->assertSame(['budi'], AksaraWriter::write('17 Budi teka')['rule']);
        // Karakter kendali U+0001 (penanda huruf besar di dalam kode) dibuang dari masukan, tidak menjadikan kata nama.
        $this->assertSame(AksaraWriter::write('aku lan budi'), AksaraWriter::write("aku lan \x01budi"));
        // Kata pertama sesudah titik, tanda tanya, tanda seru, atau baris baru juga bukan nama; sesudah koma tetap nama.
        $this->assertSame([], AksaraWriter::write("Kowe lunga? Sapa kuwi! Bapak\nSimbah")['rule']);
        $this->assertSame(['budi'], AksaraWriter::write('pak, Budi teka')['rule']);
        // Huruf kecil tidak pernah dianggap nama, juga bila katanya ada di daftar nama kamus.
        $this->assertSame(['aksara' => 'ꦲꦏꦸ ꦧꦶꦩ', 'dictionary' => [], 'rule' => []], AksaraWriter::write('aku bima'));
    }

    public function test_round_trips_through_the_latin_transliterator(): void
    {
        // Arah balik menandai pepet sebagai ê; selain itu kalimatnya kembali utuh.
        $latin = 'kula badhé tindak dhateng peken, panjenengan nitih kreta';
        $back = Transliterator::toLatin(AksaraWriter::fromLatin($latin));
        $this->assertSame('kula badhé tindak dhatêng pêkên, panjênêngan nitih krêta', $back);
        $this->assertSame($latin, str_replace('ê', 'e', $back));
    }

    public function test_agrees_with_the_dictionary_spellings(): void
    {
        // Pembanding mandiri: ejaan aksara lema kamus bahasa Jawa (ditulis penyunting Wiktionary). Angka di kelas adalah
        // yang ditampilkan halaman Terjemahan, jadi harus persis; turun berarti aturan mundur, naik berarti angka belum diperbarui.
        // Entri varian ejaan ("dated spelling of asu" untuk "asoe") tidak dihitung: ejaan Latinnya bukan ejaan baku.
        $same = $total = $variants = 0;
        $seen = [];
        $handle = gzopen(__DIR__.'/../../database/dictionaries/jv.jsonl.gz', 'rb');
        while (($line = gzgets($handle)) !== false) {
            $entry = json_decode($line, true);
            $word = $entry['word'];
            $aksara = $entry['aksara'] ?? '';
            if ($aksara === '' || str_contains($aksara, ' ') || ! preg_match('/^[\p{L}-]+$/u', $word) || isset($seen[$word."\t".$aksara])) {
                continue;
            }
            if (preg_match(AksaraWriter::SPELLING_VARIANT, $entry['glosses'][0])) {
                $variants++;

                continue;
            }
            $seen[$word."\t".$aksara] = true;
            $total++;
            $same += (int) (AksaraWriter::fromLatin($word) === $aksara);
        }
        gzclose($handle);

        $this->assertSame(AksaraWriter::DICTIONARY_AGREEMENT, [$same, $total]);
        $this->assertGreaterThan(100, $variants, 'entri varian ejaan harus dikenali, bukan ikut dihitung');
        $this->assertGreaterThan(0.97, $same / $total);
    }
}

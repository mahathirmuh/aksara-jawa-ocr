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

<?php

namespace Tests;

use Illuminate\Filesystem\Filesystem;
use Illuminate\Foundation\Testing\TestCase as BaseTestCase;

abstract class TestCase extends BaseTestCase
{
    /** Folder berkas batch terjemahan milik test ini, pengganti storage/app/mt. */
    protected ?string $mtDir = null;

    /**
     * RefreshDatabase mengosongkan database yang dipakai test. Tolak database data nyata sebelum trait itu jalan:
     * DB_DATABASE yang masih di-export di shell (mis. sesudah menguji its_aksara_test) bisa mengarahkan test ke sana.
     */
    public function createApplication()
    {
        $app = parent::createApplication();
        $config = $app['config'];
        $database = $config->get('database.connections.'.$config->get('database.default').'.database');
        if ($database === 'its_aksara') {
            throw new \RuntimeException('Test menolak database its_aksara (data nyata); pakai SQLite bawaan atau its_aksara_test.');
        }

        return $app;
    }

    protected function setUp(): void
    {
        parent::setUp();
        // Test tidak bergantung pada aset hasil `npm run build`.
        $this->withoutVite();
        // Test tidak boleh menyentuh storage/app/mt: selain database, di sana satu-satunya salinan hasil NLLB
        // (~1 jam CPU). Dulu TranslationTest menimpanya dengan angka palsu dan menghapus output.jsonl.
        $this->mtDir = sys_get_temp_dir().DIRECTORY_SEPARATOR.'aksara-mt-'.uniqid();
        config(['aksara.mt_dir' => $this->mtDir]);
    }

    protected function tearDown(): void
    {
        if ($this->mtDir) {
            (new Filesystem)->deleteDirectory($this->mtDir);
        }
        parent::tearDown();
    }
}

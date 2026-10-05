import Chart from 'chart.js/auto';

/* Warna grafik diambil dari token CSS supaya ikut tema terang/gelap. */
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const theme = () => ({
    s1: css('--series-1'), s2: css('--series-2'), grid: css('--grid'), text: css('--axis-text'),
    good: css('--good'), band: css('--band'),
});
const pct = (v, d = 1) => `${(v * 100).toFixed(d).replace('.', ',')}%`;

/* Huruf grafik sama dengan antarmuka (Inter dari app.css). */
const FONT = 'InterVariable, Inter, "Segoe UI", sans-serif';
Chart.defaults.font.family = FONT;
Chart.defaults.font.size = 12;

/* Pita latar pada sumbu x kategori, dari indeks `from` (boleh pecahan) sampai `to`. */
const bandPlugin = {
    id: 'band',
    beforeDatasetsDraw(chart, _args, opts) {
        if (opts?.from == null) return;
        const { ctx, chartArea, scales: { x } } = chart;
        const step = x.getPixelForValue(1) - x.getPixelForValue(0);
        const left = x.getPixelForValue(Math.floor(opts.from)) + step * (opts.from % 1);
        const right = Math.min(chartArea.right, x.getPixelForValue(opts.to) + step / 2);
        ctx.save();
        ctx.fillStyle = opts.color;
        ctx.fillRect(left, chartArea.top, right - left, chartArea.bottom - chartArea.top);
        ctx.fillStyle = opts.textColor;
        ctx.font = `12px ${FONT}`;
        ctx.textAlign = 'center';
        ctx.fillText(opts.label, (left + right) / 2, chartArea.top + 14);
        ctx.restore();
    },
};

/* Garis target vertikal pada sumbu nilai (grafik batang horizontal). */
const targetPlugin = {
    id: 'target',
    afterDatasetsDraw(chart, _args, opts) {
        if (opts?.value == null) return;
        const { ctx, chartArea, scales: { x } } = chart;
        const px = x.getPixelForValue(opts.value);
        ctx.save();
        ctx.strokeStyle = opts.color;
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(px, chartArea.top);
        ctx.lineTo(px, chartArea.bottom);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = opts.color;
        ctx.font = `12px ${FONT}`;
        ctx.fillText(opts.label, px + 5, chartArea.top + 12);
        ctx.restore();
    },
};

const percentAxis = (c, max = 1) => ({
    min: 0, max, grid: { color: c.grid }, border: { display: false },
    ticks: { color: c.text, callback: (v) => pct(v, 0), maxTicksLimit: 6 },
});

const builders = {
    /* Ablasi: G3 vs val berat per run k. */
    ablation(p, c) {
        return {
            type: 'line',
            data: {
                labels: p.labels,
                datasets: [
                    { label: 'G3 · 745 baris nyata', data: p.g3, borderColor: c.s1, backgroundColor: c.s1 },
                    { label: 'Val berat · 500 baris sintetis', data: p.heavy, borderColor: c.s2, backgroundColor: c.s2 },
                ].map((d) => ({ ...d, borderWidth: 2, pointRadius: 4, pointHoverRadius: 6, pointBorderColor: css('--surface'), pointBorderWidth: 2 })),
            },
            options: {
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                scales: { y: percentAxis(c), x: { grid: { display: false }, ticks: { color: c.text } } },
                plugins: {
                    legend: { labels: { color: c.text, usePointStyle: true, boxHeight: 6 } },
                    tooltip: {
                        callbacks: {
                            title: (items) => `k${items[0].dataIndex} · ${p.added[items[0].dataIndex]}`,
                            label: (item) => `${item.dataset.label.split(' · ')[0]}: ${pct(item.raw)}`,
                            afterBody: (items) => `Val bersih: ${pct(p.clean[items[0].dataIndex], 2)}`,
                        },
                    },
                    band: { from: 7.5, to: p.labels.length - 1, color: c.band, textColor: c.text, label: 'peniru pindaian' },
                },
            },
            plugins: [bandPlugin],
        };
    },

    /* Batang horizontal CER per pipeline, dengan garis target G3. */
    bars(p, c) {
        return {
            type: 'bar',
            data: { labels: p.labels, datasets: [{ data: p.values, backgroundColor: c.s1, borderRadius: 4, barThickness: 16 }] },
            options: {
                indexAxis: 'y',
                maintainAspectRatio: false,
                scales: { x: percentAxis(c), y: { grid: { display: false }, ticks: { color: c.text } } },
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { label: (item) => `CER ${pct(item.raw, 2)} · ${p.configs[item.dataIndex]}` } },
                    target: { value: 0.08, color: c.good, label: 'target G3 8%' },
                },
            },
            plugins: [targetPlugin],
        };
    },
};

/* Alpine: <div x-data="chart('ablation', data)"><canvas x-ref="canvas"></canvas></div> */
document.addEventListener('alpine:init', () => {
    window.Alpine.data('chart', (kind, payload) => ({
        instance: null,
        init() {
            this.draw();
            this.observer = new MutationObserver(() => this.draw());
            this.observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
        },
        destroy() {
            this.instance?.destroy();
            this.observer?.disconnect();
        },
        draw() {
            this.instance?.destroy();
            this.instance = new Chart(this.$refs.canvas, builders[kind](payload, theme()));
        },
    }));
});

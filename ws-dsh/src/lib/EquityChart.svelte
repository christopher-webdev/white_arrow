<script lang="ts">
  export type Pt = { t?: string; time?: string; equity: number };
  export let data: Pt[] = [];

  $: rows = (data ?? [])
    .map(d => ({ time: d.time ?? d.t, equity: Number(d.equity) }))
    .filter(d => Number.isFinite(d.equity));

  // Compute chart geometry
  const W = 1000, H = 300, PAD = 36, innerW = W - PAD * 2, innerH = H - PAD * 2;

  $: stats = (() => {
    if (rows.length < 2) {
      return { pts: [], minY: 0, maxY: 1, ticks: [] as number[] };
    }
    const ys = rows.map(r => r.equity);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const rangeY = (maxY - minY) || 1;

    const pts = rows.map((r, i) => {
      const x = PAD + (i / (rows.length - 1)) * innerW;
      const y = PAD + (1 - (r.equity - minY) / rangeY) * innerH;
      return { x, y, equity: r.equity, time: r.time };
    });

    // Y ticks ~5 lines
    const steps = 5;
    const step = rangeY / steps;
    const ticks: number[] = [];
    for (let k = 0; k <= steps; k++) ticks.push(minY + k * step);

    return { pts, minY, maxY, ticks };
  })();

  // Simple hover tooltip
  let hover: { x: number; y: number; v: number; t?: string } | null = null;

  function onMove(e: MouseEvent) {
    if (!stats.pts.length) return;
    const rect = (e.currentTarget as SVGElement).getBoundingClientRect();
    const mx = e.clientX - rect.left;
    // nearest point on X
    let best = stats.pts[0], bi = 0, bestd = Infinity;
    stats.pts.forEach((p, i) => {
      const d = Math.abs(p.x - mx);
      if (d < bestd) { best = p; bi = i; bestd = d; }
    });
    hover = { x: best.x, y: best.y, v: best.equity, t: stats.pts[bi].time };
  }
  function onLeave() { hover = null; }
</script>

<section class="rounded-2xl p-4 shadow border bg-white">
  <div class="mb-2 font-semibold">Equity Curve</div>

  {#if stats.pts.length > 1}
    <svg
      viewBox={`0 0 ${W} ${H}`} class="w-full h-80"
      on:mousemove={onMove} on:mouseleave={onLeave}
    >
      <defs>
        <!-- Gradient stroke -->
        <linearGradient id="eqStroke" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stop-color="#3b82f6" />
          <stop offset="100%" stop-color="#10b981" />
        </linearGradient>
        <!-- Gradient fill -->
        <linearGradient id="eqFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#3b82f6" stop-opacity="0.25" />
          <stop offset="100%" stop-color="#3b82f6" stop-opacity="0" />
        </linearGradient>
      </defs>

      <!-- Border -->
      <rect x="0" y="0" width={W} height={H} fill="none" stroke="rgba(0,0,0,0.08)" />

      <!-- Grid (Y) -->
      {#each stats.ticks as t}
        {@const y = PAD + (1 - (t - stats.minY) / (stats.maxY - stats.minY || 1)) * innerH}
        <line x1={PAD} x2={PAD + innerW} y1={y} y2={y} stroke="rgba(0,0,0,0.08)" />
        <text x={8} y={y + 4} font-size="11" fill="rgba(0,0,0,0.55)">
          {t.toFixed(2)}
        </text>
      {/each}

      <!-- Axes -->
      <line x1={PAD} x2={PAD} y1={PAD} y2={PAD + innerH} stroke="rgba(0,0,0,0.25)" />
      <line x1={PAD} x2={PAD + innerW} y1={PAD + innerH} y2={PAD + innerH} stroke="rgba(0,0,0,0.25)" />

      <!-- Area fill -->
      <path
        d={`M ${stats.pts[0].x} ${PAD + innerH} ` +
           stats.pts.map(p => `L ${p.x} ${p.y}`).join(' ') +
           ` L ${stats.pts.at(-1)?.x} ${PAD + innerH} Z`}
        fill="url(#eqFill)" stroke="none"
      />

      <!-- Line -->
      <polyline
        fill="none" stroke="url(#eqStroke)" stroke-width="2.5"
        points={stats.pts.map(p => `${p.x},${p.y}`).join(' ')}
      />

      <!-- Hover marker -->
      {#if hover}
        <line x1={hover.x} x2={hover.x} y1={PAD} y2={PAD + innerH} stroke="rgba(0,0,0,0.15)" />
        <circle cx={hover.x} cy={hover.y} r="4" fill="#0ea5e9" stroke="white" stroke-width="2" />
        <g transform={`translate(${Math.min(W-180, Math.max(hover.x + 8, PAD + 8))},${Math.max(PAD + 8, hover.y - 28)})`}>
          <rect width="170" height="40" rx="8" ry="8" fill="white" stroke="rgba(0,0,0,0.15)"/>
          <text x="10" y="16" font-size="12" fill="#111827">Equity: {hover.v.toFixed(2)}</text>
          {#if hover.t}<text x="10" y="30" font-size="11" fill="#6b7280">{hover.t}</text>{/if}
        </g>
      {/if}
    </svg>
  {:else}
    <div class="text-sm text-gray-500">No equity data.</div>
  {/if}
</section>

<script lang="ts">
  export let rows: Array<{
    meta_class?: number | null;
    trades?: number;
    win_rate?: number;
    profit_factor?: number | null;
    avg_R?: number | null;
    expectancy_$?: number | null;
  }> = [];

  const pct = (x?: number | null) => x == null || Number.isNaN(x) ? '—' : (x * 100).toFixed(1) + '%';
  const num = (x?: number | null, d = 2) => x == null || Number.isNaN(x) ? '—' : x.toFixed(d);
</script>

<section class="rounded-2xl p-4 shadow border bg-white overflow-auto">
  <div class="mb-2 font-semibold">Per Class</div>
  <table class="w-full text-sm">
    <thead>
      <tr class="text-left text-gray-600">
        <th class="py-2">Class</th>
        <th>Trades</th>
        <th>Win rate</th>
        <th>Profit factor</th>
        <th>Avg R</th>
        <th>Expectancy ($)</th>
      </tr>
    </thead>
    <tbody>
      {#each rows as r}
        <tr class="border-t">
          <td class="py-2">{r.meta_class ?? '—'}</td>
          <td>{r.trades ?? '—'}</td>
          <td>{pct(r.win_rate)}</td>
          <td>{num(r.profit_factor)}</td>
          <td>{num(r.avg_R)}</td>
          <td>{num(r.expectancy_$)}</td>
        </tr>
      {/each}
    </tbody>
  </table>
</section>

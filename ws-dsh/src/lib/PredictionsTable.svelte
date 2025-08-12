<script lang="ts">
  export let rows: any[] = [];

  // formatters
  const fmt = (v: any, d = 2) =>
    (v === null || v === undefined || Number.isNaN(Number(v))) ? '—' : Number(v).toFixed(d);
</script>

<section class="rounded-2xl p-4 shadow border bg-white overflow-auto">
  <div class="mb-2 font-semibold">Predictions</div>
  <table class="w-full text-sm">
    <thead>
      <tr class="text-left text-gray-600">
        <th class="py-2">Time</th>
        <th>Symbol</th>
        <th>Type</th>
        <th>clf_1_1</th>
        <th>clf_1_2</th>
        <th>reg_pred</th>
        <th>meta_class</th>
        <th>accepted</th>
      </tr>
    </thead>
    <tbody>
      {#each rows as r}
        <tr class="border-t">
          <td class="py-2">{r.timestamp ?? '—'}</td>
          <td>{r.symbol ?? '—'}</td>
          <td>{r.trade_type ?? '—'}</td>
          <td>{fmt(r.clf_1_1_prob)}</td>
          <td>{fmt(r.clf_1_2_prob)}</td>
          <td>{fmt(r.reg_pred)}</td>
          <td>{r.meta_class ?? '—'}</td>
          <td>{r.accepted ?? '—'}</td>
        </tr>
      {/each}
    </tbody>
  </table>
</section>

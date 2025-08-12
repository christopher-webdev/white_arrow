<script lang="ts">
  export type Trade = {
    id?: string;
    symbol?: string;
    trade_type?: string;
    exit_time?: string;
    profit?: number;
    R_multiple?: number | null;
    meta_class?: number | null;
  };
  export let rows: Trade[] = [];
</script>

<section class="rounded-2xl p-4 shadow border bg-white overflow-auto">
  <div class="mb-2 font-semibold">Recent Trades</div>
  <table class="w-full text-sm">
    <thead>
      <tr class="text-left text-gray-600">
        <th class="py-2">ID</th>
        <th>Symbol</th>
        <th>Type</th>
        <th>Exit</th>
        <th>PnL</th>
        <th>R</th>
        <th>Class</th>
      </tr>
    </thead>
    <tbody>
      {#each rows as r}
        <tr class="border-t">
          <td class="py-2">{r.id ?? '—'}</td>
          <td>{r.symbol ?? '—'}</td>
          <td>{r.trade_type ?? '—'}</td>
          <td>{r.exit_time ?? '—'}</td>
          <td>{r.profit ?? '—'}</td>
          <td>{r.R_multiple ?? '—'}</td>
          <td>{r.meta_class ?? '—'}</td>
        </tr>
      {/each}
    </tbody>
  </table>
</section>

<script lang="ts">
  export type Summary = {
    trades?: number;
    win_rate?: number;
    profit_factor?: number | null;
    net_profit?: number;
    cagr?: number | null;
    sharpe_annual?: number | null;
    max_drawdown_pct?: number | null;
    expectancy_R?: number | null;
    payoff_ratio?: number | null;
  };

  export let summary: Summary = {};

  const pct = (x?: number | null) =>
    (x === null || x === undefined || Number.isNaN(x)) ? '—' : (x * 100).toFixed(1) + '%';
  const num = (x?: number | null, d = 2) =>
    (x === null || x === undefined || Number.isNaN(x)) ? '—' : x.toFixed(d);
</script>

<div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
  <div class="rounded-2xl p-4 shadow border bg-white">
    <div class="text-sm text-gray-500">Trades</div>
    <div class="text-2xl font-semibold">{summary.trades ?? '—'}</div>
  </div>
  <div class="rounded-2xl p-4 shadow border bg-white">
    <div class="text-sm text-gray-500">Win rate</div>
    <div class="text-2xl font-semibold">{pct(summary.win_rate)}</div>
  </div>
  <div class="rounded-2xl p-4 shadow border bg-white">
    <div class="text-sm text-gray-500">Profit factor</div>
    <div class="text-2xl font-semibold">{num(summary.profit_factor)}</div>
  </div>
  <div class="rounded-2xl p-4 shadow border bg-white">
    <div class="text-sm text-gray-500">Net profit</div>
    <div class="text-2xl font-semibold">{num(summary.net_profit)}</div>
  </div>
  <div class="rounded-2xl p-4 shadow border bg-white">
    <div class="text-sm text-gray-500">CAGR</div>
    <div class="text-2xl font-semibold">{pct(summary.cagr)}</div>
  </div>
  <div class="rounded-2xl p-4 shadow border bg-white">
    <div class="text-sm text-gray-500">Sharpe (annual)</div>
    <div class="text-2xl font-semibold">{num(summary.sharpe_annual)}</div>
  </div>
  <div class="rounded-2xl p-4 shadow border bg-white">
    <div class="text-sm text-gray-500">Max drawdown</div>
    <div class="text-2xl font-semibold">{pct(summary.max_drawdown_pct)}</div>
  </div>
  <div class="rounded-2xl p-4 shadow border bg-white">
    <div class="text-sm text-gray-500">Expectancy (R)</div>
    <div class="text-2xl font-semibold">{num(summary.expectancy_R)}</div>
  </div>
  <div class="rounded-2xl p-4 shadow border bg-white">
    <div class="text-sm text-gray-500">Payoff ratio</div>
    <div class="text-2xl font-semibold">{num(summary.payoff_ratio)}</div>
  </div>
</div>

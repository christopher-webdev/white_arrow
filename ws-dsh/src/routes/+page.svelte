<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '$lib/api';
  import { settings, buildQuery } from '$lib/stores';
  import { get } from 'svelte/store';
  import SummaryCards from '$lib/SummaryCards.svelte';
  import EquityChart from '$lib/EquityChart.svelte';
  import PerClassTable from '$lib/PerClassTable.svelte';
  import TradesTable from '$lib/TradesTable.svelte';

  let s = get(settings);
  let full: any = {};
  let loading = true;
  let err: string | null = null;

  $: summary = full?.summary ?? {};

  async function load() {
    loading = true; err = null;
    try {
      full = await api.full(buildQuery(s));
    } catch (e: any) {
      err = e?.message ?? String(e);
    } finally {
      loading = false;
    }
  }

  const unsub = settings.subscribe((v) => { s = v; load(); });
  onMount(load);
</script>

{#if loading}
  <div class="text-sm text-gray-500">Loading…</div>
{:else if err}
  <div class="text-sm text-red-600">Error: {err}</div>
{:else}
  <div class="grid gap-6">
    
    <SummaryCards {summary} />
    <EquityChart data={full.equity_curve ?? []} />
    <PerClassTable rows={full.per_class ?? []} />
    <a href="/predictions" class="inline-block px-4 py-2 rounded-xl border">View Predictions</a>
    <TradesTable rows={full.trades_table ?? []} />

  </div>
{/if}

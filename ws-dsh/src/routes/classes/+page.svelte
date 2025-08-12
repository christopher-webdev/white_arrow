<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '$lib/api';
  import { settings, buildQuery } from '$lib/stores';
  import { get } from 'svelte/store';
  import PerClassTable from '$lib/PerClassTable.svelte';
  import TradesTable from '$lib/TradesTable.svelte';
  import SummaryCards from '$lib/SummaryCards.svelte';

  let s = get(settings);

  let perClass: any[] = [];
  let loading = true;
  let err: string | null = null;

  let selectedClass: number | null = null;
  let classLoading = false;
  let classErr: string | null = null;
  let classData: any = null; // full metrics for selected class (summary + trades)

  async function loadPerClass() {
    loading = true; err = null;
    try {
      const q = buildQuery(s);
      const full = await api.full(q);
      perClass = full?.per_class ?? [];
    } catch (e: any) {
      err = e?.message ?? String(e);
    } finally {
      loading = false;
    }
  }

  async function loadClass(c: number) {
    selectedClass = c;
    classLoading = true; classErr = null; classData = null;
    try {
      // force only_classes to this selection, keep threshold/includeReject
      const q = buildQuery({ ...s, onlyClasses: String(c) });
      classData = await api.full(q);
    } catch (e: any) {
      classErr = e?.message ?? String(e);
    } finally {
      classLoading = false;
    }
  }

  // reload when settings change
  const unsub = settings.subscribe((v) => {
    s = v;
    loadPerClass();
    if (selectedClass != null) loadClass(selectedClass);
  });

  onMount(loadPerClass);

  const pct0 = (x?: number) => Math.round((x ?? 0) * 100);
</script>

<h1 class="text-xl font-semibold mb-4">Classes</h1>

{#if loading}
  <div class="text-sm text-gray-500">Loading…</div>
{:else if err}
  <div class="text-sm text-red-600">Error: {err}</div>
{:else}
  <PerClassTable rows={perClass} />
  <div class="text-sm text-gray-600 mt-2">Click a row to drill down into trades for that class.</div>

  <div class="mt-4 rounded-2xl p-4 border bg-white overflow-auto">
    <table class="w-full text-sm">
      <thead>
        <tr class="text-left text-gray-600">
          <th class="py-2">Class</th>
          <th>Trades</th>
          <th>Win rate</th>
          <th>Profit factor</th>
          <th>Avg R</th>
          <th>Expectancy ($)</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {#each perClass as r}
          <tr class="border-t hover:bg-gray-50">
            <td class="py-2">{r.meta_class ?? '—'}</td>
            <td>{r.trades ?? '—'}</td>
            <td>{pct0(r.win_rate)}%</td>
            <td>{r.profit_factor ?? '—'}</td>
            <td>{r.avg_R ?? '—'}</td>
            <td>{r.expectancy_$ ?? '—'}</td>
            <td>
              <button class="px-3 py-1 rounded-lg border" on:click={() => loadClass(r.meta_class)}>View</button>
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>

  {#if selectedClass != null}
    <h2 class="text-lg font-semibold mt-6">Class {selectedClass} details</h2>
    {#if classLoading}
      <div class="text-sm text-gray-500">Loading class…</div>
    {:else if classErr}
      <div class="text-sm text-red-600">Error: {classErr}</div>
    {:else if classData}
      <div class="grid gap-4 mt-2">
        <SummaryCards summary={classData.summary ?? {}} />
        <TradesTable rows={classData.trades_table ?? []} />
      </div>
    {/if}
  {/if}
{/if}

<script lang="ts">
  import { api } from '$lib/api';
  import { onMount } from 'svelte';
  import TradesTable from '$lib/TradesTable.svelte';
  import { settings, buildQuery } from '$lib/stores';
  import { get } from 'svelte/store';

  let s = get(settings);
  let rows: any[] = [];
  let loading = true;
  let err: string | null = null;

  async function load() {
    loading = true; err = null;
    try {
      const full = await api.full(buildQuery(s));
      rows = full?.trades_table ?? [];
    } catch (e: any) {
      err = e?.message ?? String(e);
    } finally {
      loading = false;
    }
  }

  const unsub = settings.subscribe((v) => { s = v; load(); });
  onMount(load);
</script>

<h1 class="text-xl font-semibold mb-4">Trades</h1>
{#if loading}
  <div class="text-sm text-gray-500">Loading…</div>
{:else if err}
  <div class="text-sm text-red-600">Error: {err}</div>
{:else}
  <TradesTable {rows} />
{/if}

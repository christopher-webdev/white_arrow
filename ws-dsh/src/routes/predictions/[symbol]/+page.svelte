<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { api } from '$lib/api';
  import SymbolList from '$lib/SymbolList.svelte';
  import PredictionsTable from '$lib/PredictionsTable.svelte';

  let symbol = '';
  let symbols: string[] = [];
  let loadingSymbols = true;
  let errSymbols: string | null = null;

  let rows: any[] = [];
  let loading = true;
  let err: string | null = null;

  let limit = 200;

  async function loadSymbols() {
    loadingSymbols = true; errSymbols = null;
    try {
      const res = await api.symbols();
      symbols = res.symbols ?? [];
    } catch (e: any) {
      errSymbols = e?.message ?? String(e);
    } finally {
      loadingSymbols = false;
    }
  }

  async function loadPreds() {
    loading = true; err = null;
    try {
      rows = await api.predictions(symbol, limit);
    } catch (e: any) {
      err = e?.message ?? String(e);
    } finally {
      loading = false;
    }
  }

  $: symbol = $page.params.symbol ? decodeURIComponent($page.params.symbol) : '';

  // initial loads
  onMount(async () => {
    await Promise.all([loadSymbols(), loadPreds()]);
  });

  // reload when symbol or limit changes
  $: if (symbol) {
    // debounce-ish: trigger loadPreds when symbol or limit changes
    loadPreds();
  }
</script>

<h1 class="text-xl font-semibold mb-4">Predictions – {symbol}</h1>

<div class="grid grid-cols-1 md:grid-cols-4 gap-4">
  <div class="md:col-span-1">
    {#if loadingSymbols}
      <div class="text-sm text-gray-500">Loading symbols…</div>
    {:else if errSymbols}
      <div class="text-sm text-red-600">Error: {errSymbols}</div>
    {:else}
      <SymbolList {symbols} active={symbol} />
    {/if}
  </div>

  <div class="md:col-span-3 space-y-4">
    <section class="rounded-2xl p-4 shadow border bg-white">
      <div class="flex items-center gap-3">
        <label class="text-sm text-gray-600">Limit</label>
        <input
          type="number" min="1" max="5000"
          class="border rounded-xl px-3 py-2 w-28"
          bind:value={limit}
        />
        <button class="px-3 py-2 rounded-xl border" on:click={loadPreds}>Reload</button>
      </div>
    </section>

    {#if loading}
      <div class="text-sm text-gray-500">Loading predictions…</div>
    {:else if err}
      <div class="text-sm text-red-600">Error: {err}</div>
    {:else}
      <PredictionsTable rows={rows} />
    {/if}
  </div>
</div>

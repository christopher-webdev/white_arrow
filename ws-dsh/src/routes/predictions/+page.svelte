<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '$lib/api';
  import SymbolList from '$lib/SymbolList.svelte';

  let loading = true;
  let err: string | null = null;
  let symbols: string[] = [];

  onMount(async () => {
    try {
      const res = await api.symbols();
      symbols = res.symbols ?? [];
    } catch (e: any) {
      err = e?.message ?? String(e);
    } finally {
      loading = false;
    }
  });
</script>

<h1 class="text-xl font-semibold mb-4">Predictions</h1>

{#if loading}
  <div class="text-sm text-gray-500">Loading symbols…</div>
{:else if err}
  <div class="text-sm text-red-600">Error: {err}</div>
{:else}
  <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
    <div class="md:col-span-1">
      <SymbolList {symbols} active={null} />
    </div>
    <div class="md:col-span-3">
      <section class="rounded-2xl p-6 shadow border bg-white">
        <div class="text-sm text-gray-600">
          Select a symbol on the left to view recent predictions.
        </div>
      </section>
    </div>
  </div>
{/if}

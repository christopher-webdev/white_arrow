<script lang="ts">
  export let symbols: string[] = [];
  export let active: string | null = null;
  export let baseHref = '/predictions';
</script>

<nav class="rounded-2xl border bg-white p-3 shadow">
  <div class="text-sm text-gray-500 mb-2">Symbols</div>
  <ul class="space-y-1">
    {#each symbols as s}
      <li>
        <a
          href={`${baseHref}/${encodeURIComponent(s)}`}
          class="block px-3 py-2 rounded-lg hover:bg-gray-100"
          class:bg-gray-100={active === s}
          class:font-medium={active === s}
        >
          {s}
        </a>
      </li>
    {/each}
    {#if !symbols.length}
      <li class="text-sm text-gray-500 px-3 py-2">No symbols found.</li>
    {/if}
  </ul>
</nav>

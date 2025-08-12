<script lang="ts">
  import "../app.css";
  import { page } from '$app/stores';

  const nav = [
    { href: '/', label: 'Overview' },
    { href: '/trades', label: 'Trades' },
    { href: '/classes', label: 'Classes' },
	{ href: '/predictions', label: 'Predictions' },
    { href: '/settings', label: 'Settings' }
  ];
</script>

<div class="min-h-screen bg-gray-50">
  <header class="sticky top-0 z-10 bg-white border-b">
    <div class="mx-auto max-w-6xl px-4 py-3 flex items-center gap-6">
      <a href="/" class="font-semibold">White Arrow Trading</a>

      <nav class="flex items-center gap-2 text-sm">
        {#each nav as n}
          <a
            href={n.href}
            class="px-3 py-2 rounded-xl hover:bg-gray-100"
            class:bg-gray-100={$page.url.pathname === n.href}
            class:font-medium={$page.url.pathname === n.href}
          >
            {n.label}
          </a>
        {/each}
      </nav>
    </div>
  </header>

  <main class="mx-auto max-w-6xl px-4 py-6">
    <slot />
  </main>
</div>


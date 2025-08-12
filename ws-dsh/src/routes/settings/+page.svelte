<script lang="ts">
  import { settings, buildQuery } from '$lib/stores';
  import { get } from 'svelte/store';
  let s = get(settings);

  function save() {
    settings.set(s);
  }
  function reset() {
    s = { threshold: 0, includeReject: false, onlyClasses: '' };
    settings.set(s);
  }
</script>

<h1 class="text-xl font-semibold mb-4">Settings</h1>

<section class="rounded-2xl p-4 shadow border bg-white grid gap-4">
  <div>
    <label class="text-sm text-gray-600">Meta prob threshold: <span class="font-medium">{s.threshold}</span></label>
    <input type="range" min="0" max="1" step="0.05" bind:value={s.threshold} class="w-full"/>
    <p class="text-xs text-gray-500 mt-1">Filter trades by meta_max_prob ≥ threshold.</p>
  </div>

  <label class="flex items-center gap-2">
    <input type="checkbox" bind:checked={s.includeReject}/>
    <span>Include rejected class (meta_class = 0)</span>
  </label>

  <div>
    <label class="text-sm text-gray-600">Only classes (comma-separated)</label>
    <input
      class="mt-1 w-full border rounded-xl px-3 py-2"
      placeholder="e.g. 1,2"
      bind:value={s.onlyClasses}
    />
    <p class="text-xs text-gray-500 mt-1">Leave empty to include all classes (except rejects unless checked).</p>
  </div>

  <div class="flex gap-3">
    <button class="px-4 py-2 rounded-xl bg-black text-white" on:click={save}>Save</button>
    <button class="px-4 py-2 rounded-xl border" on:click={reset}>Reset</button>
  </div>

  <div class="text-xs text-gray-500">
    Current query: <code class="px-2 py-1 rounded bg-gray-100">{buildQuery(s) || '(none)'}</code>
  </div>
</section>

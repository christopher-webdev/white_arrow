<script lang="ts">
  export let title: string | undefined = undefined;
  export let className = '';
</script>

<section class={"rounded-2xl p-4 shadow border bg-white " + className}>
  {#if title}<h2 class="mb-2 font-semibold">{title}</h2>{/if}
  <slot />
</section>

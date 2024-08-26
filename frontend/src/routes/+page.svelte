<h1>Typstifier</h1>

<script lang="ts">
  import { infer_symbol } from "$lib/pkg/inference";
  let arr = new Array(64 * 64).fill(0);
  console.log(infer_symbol(arr));
</script>

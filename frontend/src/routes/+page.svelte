<h1>Typstifier</h1>

<script lang="ts">
  import { infer_symbol } from "$lib/pkg/inference";
  let arr = [1.0, 2.0, 3.0, 4.0, 5.0];
  console.log(infer_symbol(arr));
</script>

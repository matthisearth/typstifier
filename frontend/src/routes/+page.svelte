<script lang="ts">
  import { infer_symbol } from "$lib/pkg/inference";
  import { onMount } from "svelte";
  import names from "$lib/names.json";

  let canvas: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D;
  let drawing = false;

  const lineWidth = 8;
  const canvasSize = 256;
  const drawSize = 64;
  const maxNumShow = 10;

  let pixelArray = new Float32Array(drawSize * drawSize).fill(0.0);
  let rankedSymbols: Uint32Array = new Uint32Array(0);

  function startDrawing(event: MouseEvent | TouchEvent) {
    event.preventDefault();
    drawing = true;
    continueDrawing(event);
  }

  function stopDrawing(event: MouseEvent | TouchEvent | FocusEvent) {
    event.preventDefault();
    drawing = false;
    ctx.beginPath();
  }

  function continueDrawing(event: MouseEvent | TouchEvent) {
    event.preventDefault();
    if (!drawing) return;

    // Get the canvas boundaries and mouse position
    const rect = canvas.getBoundingClientRect();
    let x = 0.0;
    let y = 0.0;

    if (event instanceof MouseEvent) {
      x = event.clientX - rect.left;
      y = event.clientY - rect.top;
    }

    if (event instanceof TouchEvent) {
      x = event.touches[0].clientX - rect.left;
      y = event.touches[0].clientY - rect.top;
    }

    ctx.lineWidth = lineWidth;
    ctx.lineCap = "round";
    ctx.strokeStyle = "#000";

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);

    storeBrushPixels(x, y);
  }

  function clearDrawing() {
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    pixelArray = new Float32Array(drawSize * drawSize).fill(0.0);
    evaluateDrawing();
  }

  function evaluateDrawing() {
    if (pixelArray.every((item) => item == 0.0)) {
      rankedSymbols = new Uint32Array(0);
    } else {
      rankedSymbols = infer_symbol(pixelArray);
    }
  }

  function storeBrushPixels(x: number, y: number) {
    const radius = lineWidth / 2;

    for (let offsetX = -radius; offsetX <= radius; offsetX++) {
      for (let offsetY = -radius; offsetY <= radius; offsetY++) {
        const distance = Math.sqrt(offsetX * offsetX + offsetY * offsetY);
        const pixelX = Math.round(((x + offsetX) * drawSize) / canvasSize);
        const pixelY = Math.round(((y + offsetY) * drawSize) / canvasSize);
        if (
          pixelX >= 0 &&
          pixelX < drawSize &&
          pixelY >= 0 &&
          pixelY < drawSize &&
          distance <= radius
        ) {
          pixelArray[pixelX + drawSize * pixelY] = 1.0;
        }
      }
    }
  }

  onMount(() => {
    const ctxLocal = canvas.getContext("2d");
    if (ctxLocal instanceof CanvasRenderingContext2D) {
      ctx = ctxLocal;
    }

    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  });
</script>

<div class="flex-grow w-full max-w-screen-lg mx-auto object-contain px-5 pt-2 pb-5 bg-gray-200">
  <h1 class="text-3xl font-bold mt-4 mb-6 font-mono">Typstifier</h1>

  <canvas
    class="rounded-lg"
    bind:this={canvas}
    width={canvasSize}
    height={canvasSize}
    on:mousedown={startDrawing}
    on:touchstart={startDrawing}
    on:mouseup={stopDrawing}
    on:touchend={stopDrawing}
    on:touchcancel={stopDrawing}
    on:mouseout={stopDrawing}
    on:blur={stopDrawing}
    on:mousemove={continueDrawing}
    on:touchmove={continueDrawing}
  ></canvas>

  <div class="my-2">
    <button
      class="font-mono bg-red-500 text-white rounded px-4 py-2 m-2 hover:bg-red-600 active:bg-red-700"
      on:click={() => clearDrawing()}>Clear</button>
    <button
      class="font-mono bg-green-500 text-white rounded px-4 py-2 m-2 hover:bg-green-600 active:bg-green-700"
      on:click={() => evaluateDrawing()}>Evaluate</button>
  </div>

  <div class="flex flex-row flex-wrap">
    {#each rankedSymbols.slice(0, maxNumShow) as symbol}
      <div class="bg-white p-6 rounded-lg m-4 flex flex-col items-center">
        <img
          class="mb-6"
          src="symbols/{symbol}.png"
          alt={names[symbol].sym_name}
        />
        <span class="font-mono text-center">{names[symbol].sym_name}</span>
      </div>
    {/each}
  </div>
</div>

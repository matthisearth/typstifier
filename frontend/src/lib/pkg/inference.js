
import * as wasm from "./inference_bg.wasm";
import { __wbg_set_wasm } from "./inference_bg.js";
__wbg_set_wasm(wasm);
export * from "./inference_bg.js";

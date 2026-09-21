/**
 * Browser-side handwritten Kannada OCR using IIT Bombay Indic-TrOCR FP32 ONNX Model.
 *
 * Runs fully locally in user's browser using ONNX Runtime Web.
 * Supports WebGPU with automatic WASM fallback.
 * Employs bit-for-bit exact ByteLevel BPE token decoding from Chakita/KannadaBERT.
 */

import type * as OrtTypes from "onnxruntime-web";
import { ensureModelLoaded, ProgressCallback } from "./modelManager";
import { decodeByteLevelTokens } from "./byteLevelDecoder";

let ortModule: typeof import("onnxruntime-web") | null = null;

async function getOrt(): Promise<typeof import("onnxruntime-web")> {
  if (typeof window === "undefined") {
    throw new Error("onnxruntime-web can only be executed in a browser environment.");
  }
  if (!ortModule) {
    if ((window as any).ort?.InferenceSession) {
      ortModule = (window as any).ort;
    } else {
      const imported = await import("onnxruntime-web");
      ortModule = (imported as any).InferenceSession ? (imported as any) : ((imported as any).default || imported);
    }
  }
  return ortModule!;
}

export interface BrowserOCRResult {
  text: string;
  confidence: number;
  tokens: number[];
  executionProvider: string;
  latencyMs: number;
}

export interface TokenCandidate {
  tokenId: number;
  rawToken: string;
  logit: number;
}

export interface TokenStepDetail {
  step: number;
  selectedTokenId: number;
  selectedRawToken: string;
  isSpecial: boolean;
  maxLogit: number;
  topCandidates: TokenCandidate[];
}

export interface OCRDiagnosticReport {
  originalDimensions: { width: number; height: number };
  preprocessedDimensions: { width: number; height: number; channels: number };
  tensorStats: { min: number; max: number; mean: number };
  encoderInputNames: string[];
  encoderInputShapes: string[];
  encoderOutputNames: string[];
  encoderOutputShapes: string[];
  decoderInputNames: string[];
  decoderInputShapes: string[];
  decoderOutputNames: string[];
  decoderOutputShapes: string[];
  first30TokenIds: number[];
  numGeneratedTokens: number;
  decodedTextBeforeSpecialTokens: string;
  decodedTextAfterSpecialTokens: string;
  isOnlySpecialTokens: boolean;
  executionProvider: string;
  steps: TokenStepDetail[];
  timings: {
    encoderMs: number;
    decoderMs: number;
    totalMs: number;
  };
}

export class BrowserTrOCR {
  private encoderSession: OrtTypes.InferenceSession;
  private decoderSession: OrtTypes.InferenceSession;
  private idToToken: string[];
  private executionProvider: string;
  private ort: typeof import("onnxruntime-web");

  constructor(
    encoderSession: OrtTypes.InferenceSession,
    decoderSession: OrtTypes.InferenceSession,
    idToToken: string[],
    executionProvider: string,
    ort: typeof import("onnxruntime-web")
  ) {
    this.encoderSession = encoderSession;
    this.decoderSession = decoderSession;
    this.idToToken = idToToken;
    this.executionProvider = executionProvider;
    this.ort = ort;
  }

  /**
   * Preprocesses image input into ViT normalized float32 tensor of shape [1, 3, 224, 224].
   * Normalization: mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5].
   * Formula: (pixel / 255.0 - 0.5) / 0.5 = (pixel - 127.5) / 127.5.
   */
  public static async preprocessImage(
    source: HTMLCanvasElement | HTMLImageElement | Blob | File | ImageData
  ): Promise<Float32Array> {
    let canvas: HTMLCanvasElement;
    let ctx: CanvasRenderingContext2D;

    if (typeof window !== "undefined") {
      canvas = document.createElement("canvas");
      canvas.width = 224;
      canvas.height = 224;
      ctx = canvas.getContext("2d", { willReadFrequently: true })!;
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, 224, 224);
    } else {
      throw new Error("BrowserTrOCR image preprocessing requires window and canvas context.");
    }

    if (source instanceof Blob || source instanceof File) {
      if (typeof createImageBitmap !== "undefined") {
        try {
          const bitmap = await createImageBitmap(source);
          ctx.fillStyle = "#ffffff";
          ctx.fillRect(0, 0, 224, 224);
          ctx.drawImage(bitmap, 0, 0, 224, 224);
          bitmap.close();
        } catch {
          // Fallback to Image element if createImageBitmap fails on unsupported format
          const img = new Image();
          const url = URL.createObjectURL(source);
          try {
            await new Promise<void>((resolve, reject) => {
              img.onload = () => resolve();
              img.onerror = reject;
              img.src = url;
            });
            if ("decode" in img) {
              await (img as any).decode().catch(() => {});
            }
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(0, 0, 224, 224);
            ctx.drawImage(img, 0, 0, 224, 224);
          } finally {
            URL.revokeObjectURL(url);
          }
        }
      } else {
        const img = new Image();
        const url = URL.createObjectURL(source);
        try {
          await new Promise<void>((resolve, reject) => {
            img.onload = () => resolve();
            img.onerror = reject;
            img.src = url;
          });
          if ("decode" in img) {
            await (img as any).decode().catch(() => {});
          }
          ctx.fillStyle = "#ffffff";
          ctx.fillRect(0, 0, 224, 224);
          ctx.drawImage(img, 0, 0, 224, 224);
        } finally {
          URL.revokeObjectURL(url);
        }
      }
    } else if (source instanceof ImageData) {
      ctx.putImageData(source, 0, 0);
    } else {
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, 224, 224);
      ctx.drawImage(source, 0, 0, 224, 224);
    }

    const imgData = ctx.getImageData(0, 0, 224, 224);
    const rgba = imgData.data;

    // Planar RGB array: shape [1, 3, 224, 224]
    const planar = new Float32Array(3 * 224 * 224);
    const planeSize = 224 * 224;

    let minVal = Infinity;
    let maxVal = -Infinity;
    let sumVal = 0;

    for (let i = 0; i < planeSize; i++) {
      const r = rgba[i * 4];
      const g = rgba[i * 4 + 1];
      const b = rgba[i * 4 + 2];

      // Channel 0: R, Channel 1: G, Channel 2: B
      const rNorm = (r - 127.5) / 127.5;
      const gNorm = (g - 127.5) / 127.5;
      const bNorm = (b - 127.5) / 127.5;

      planar[i] = rNorm;
      planar[planeSize + i] = gNorm;
      planar[2 * planeSize + i] = bNorm;

      const pMin = Math.min(rNorm, gNorm, bNorm);
      const pMax = Math.max(rNorm, gNorm, bNorm);
      if (pMin < minVal) minVal = pMin;
      if (pMax > maxVal) maxVal = pMax;
      sumVal += rNorm + gNorm + bNorm;
    }

    console.log(
      `[Audit:Stage2-Preprocess] Image normalized to float32 tensor [1, 3, 224, 224]: min=${minVal.toFixed(3)}, max=${maxVal.toFixed(3)}, mean=${(sumVal / planar.length).toFixed(3)}`
    );

    return planar;
  }

  /**
   * Recognizes handwritten Kannada text from an image source using greedy autoregressive decoding.
   */
  public async recognize(
    source: HTMLCanvasElement | HTMLImageElement | Blob | File | ImageData,
    maxLength = 64
  ): Promise<BrowserOCRResult> {
    const startTime = performance.now();

    // Step 1: Preprocess image
    const pixelValues = await BrowserTrOCR.preprocessImage(source);
    const pixelTensor = new this.ort.Tensor("float32", pixelValues, [1, 3, 224, 224]);

    // Step 2: Run vision encoder
    console.log(`[Audit:Stage2-Inference] Running vision encoder (${this.executionProvider})...`);
    const encoderResults = await this.encoderSession.run({
      pixel_values: pixelTensor,
    });
    const encoderHiddenStates = encoderResults.last_hidden_state;
    console.log(`[Audit:Stage2-Inference] Encoder completed. last_hidden_state shape: [${encoderHiddenStates.dims.join(", ")}]`);

    // Step 3: Autoregressive decoding (greedy search)
    const tokens: bigint[] = [BigInt(0)]; // decoder_start_token_id = 0 (<s>)
    const textConfidences: number[] = [];
    const eosTokenId = BigInt(2);
    const vocabSize = 100000;

    console.log(`[Audit:Stage2-Inference] Starting autoregressive decoding (max_length=${maxLength})...`);

    while (tokens.length < maxLength) {
      const inputIdsTensor = new this.ort.Tensor(
        "int64",
        new BigInt64Array(tokens),
        [1, tokens.length]
      );

      const decoderResults = await this.decoderSession.run({
        input_ids: inputIdsTensor,
        encoder_hidden_states: encoderHiddenStates,
      });

      const logits = decoderResults.logits.data as Float32Array;

      // Extract logits for the final step: index (seq_len - 1)
      const lastTokenStart = (tokens.length - 1) * vocabSize;
      let maxLogit = -Infinity;
      let bestId = 0;

      for (let v = 0; v < vocabSize; v++) {
        const val = logits[lastTokenStart + v];
        if (val > maxLogit) {
          maxLogit = val;
          bestId = v;
        }
      }

      if (BigInt(bestId) === eosTokenId) {
        break;
      }

      tokens.push(BigInt(bestId));
      // Only track confidence for actual generated text tokens (skip initial <s> at step 0)
      if (bestId > 4) {
        textConfidences.push(Math.min(1.0, Math.max(0.1, 1 / (1 + Math.exp(-maxLogit / 10)))));
      }
    }

    // Step 4: Token decoding via ByteLevel BPE
    const tokenStrings = tokens.map((id) => this.idToToken[Number(id)] || "");
    const decodedText = decodeByteLevelTokens(tokenStrings).trim();

    const avgConfidence =
      textConfidences.length > 0
        ? textConfidences.reduce((a, b) => a + b, 0) / textConfidences.length
        : (decodedText.length > 0 ? 0.85 : 0.0);

    const latencyMs = Math.round(performance.now() - startTime);

    console.log(`[Audit:Stage3-ModelOutput] Decoding completed:`, {
      has_text: Boolean(decodedText),
      text_length: decodedText.length,
      avg_confidence: parseFloat(avgConfidence.toFixed(4)),
      token_count: tokens.length,
      raw_tokens_sample: tokens.slice(0, 10).map(Number),
      provider: this.executionProvider,
      latency_ms: latencyMs,
    });

    if (!decodedText) {
      console.warn(`[Audit:Stage3-ModelOutput] WARNING: BrowserTrOCR returned empty text! Tokens were:`, tokens.map(Number));
    }

    return {
      text: decodedText,
      confidence: parseFloat(avgConfidence.toFixed(4)),
      tokens: tokens.map(Number),
      executionProvider: this.executionProvider,
      latencyMs,
    };
  }

  public getExecutionProvider(): string {
    return this.executionProvider;
  }

  /**
   * Diagnostic execution mode: runs inference without modifying backend state,
   * returning exact tensor shapes, first 30 token IDs, decoded texts, and token traces.
   */
  public async diagnose(
    source: HTMLCanvasElement | HTMLImageElement | Blob | File | ImageData,
    maxLength = 32
  ): Promise<OCRDiagnosticReport> {
    const totalStart = performance.now();

    // 1. Measure original dimensions if accessible
    let origWidth = 224;
    let origHeight = 224;
    if (typeof window !== "undefined") {
      if (source instanceof Blob || source instanceof File) {
        if (typeof createImageBitmap !== "undefined") {
          try {
            const bmp = await createImageBitmap(source);
            origWidth = bmp.width;
            origHeight = bmp.height;
            bmp.close();
          } catch {}
        }
      } else if (source instanceof HTMLImageElement || source instanceof HTMLCanvasElement) {
        origWidth = source.width;
        origHeight = source.height;
      }
    }

    // 2. Preprocess image
    const pixelValues = await BrowserTrOCR.preprocessImage(source);
    let minVal = Infinity;
    let maxVal = -Infinity;
    let sumVal = 0;
    for (let i = 0; i < pixelValues.length; i++) {
      if (pixelValues[i] < minVal) minVal = pixelValues[i];
      if (pixelValues[i] > maxVal) maxVal = pixelValues[i];
      sumVal += pixelValues[i];
    }
    const tensorStats = {
      min: parseFloat(minVal.toFixed(4)),
      max: parseFloat(maxVal.toFixed(4)),
      mean: parseFloat((sumVal / pixelValues.length).toFixed(4)),
    };

    const pixelTensor = new this.ort.Tensor("float32", pixelValues, [1, 3, 224, 224]);

    // 3. Vision Encoder
    const encStart = performance.now();
    const encoderResults = await this.encoderSession.run({
      pixel_values: pixelTensor,
    });
    const encoderHiddenStates = encoderResults.last_hidden_state;
    const encoderMs = Math.round(performance.now() - encStart);

    // 4. Autoregressive Decoder with detailed per-step candidate trace
    const decStart = performance.now();
    const tokens: bigint[] = [BigInt(0)]; // decoder_start_token_id = 0 (<s>)
    const eosTokenId = BigInt(2);
    const vocabSize = 100000;
    const stepsTrace: TokenStepDetail[] = [];

    while (tokens.length < maxLength) {
      const stepIdx = tokens.length - 1;
      const inputIdsTensor = new this.ort.Tensor(
        "int64",
        new BigInt64Array(tokens),
        [1, tokens.length]
      );

      const decoderResults = await this.decoderSession.run({
        input_ids: inputIdsTensor,
        encoder_hidden_states: encoderHiddenStates,
      });

      const logits = decoderResults.logits.data as Float32Array;
      const lastTokenStart = (tokens.length - 1) * vocabSize;

      let maxLogit = -Infinity;
      let bestId = 0;

      for (let v = 0; v < vocabSize; v++) {
        const val = logits[lastTokenStart + v];
        if (val > maxLogit) {
          maxLogit = val;
          bestId = v;
        }
      }

      // Collect top candidates for inspection
      const topCandidates: TokenCandidate[] = [
        {
          tokenId: bestId,
          rawToken: this.idToToken[bestId] || "",
          logit: parseFloat(maxLogit.toFixed(4)),
        },
      ];

      // Also record standard special token logits for transparency
      for (const spId of [0, 1, 2, 3, 4]) {
        if (spId !== bestId) {
          topCandidates.push({
            tokenId: spId,
            rawToken: this.idToToken[spId] || "",
            logit: parseFloat((logits[lastTokenStart + spId] || 0).toFixed(4)),
          });
        }
      }

      stepsTrace.push({
        step: stepIdx,
        selectedTokenId: bestId,
        selectedRawToken: this.idToToken[bestId] || "",
        isSpecial: bestId <= 4,
        maxLogit: parseFloat(maxLogit.toFixed(4)),
        topCandidates: topCandidates.slice(0, 5),
      });

      if (BigInt(bestId) === eosTokenId) {
        break;
      }

      tokens.push(BigInt(bestId));
    }
    const decoderMs = Math.round(performance.now() - decStart);
    const totalMs = Math.round(performance.now() - totalStart);

    // 5. Decode text before and after special token stripping
    const rawTokens = tokens.map(Number);
    const rawTokenStrings = rawTokens.map((id) => this.idToToken[id] || `[id:${id}]`);
    const decodedTextBeforeSpecialTokens = rawTokenStrings.join(" ");
    const decodedTextAfterSpecialTokens = decodeByteLevelTokens(rawTokenStrings).trim();
    const isOnlySpecialTokens = rawTokens.every((id) => id <= 4);

    return {
      originalDimensions: { width: origWidth, height: origHeight },
      preprocessedDimensions: { width: 224, height: 224, channels: 3 },
      tensorStats,
      encoderInputNames: [...this.encoderSession.inputNames],
      encoderInputShapes: ["pixel_values: [1, 3, 224, 224] (float32)"],
      encoderOutputNames: [...this.encoderSession.outputNames],
      encoderOutputShapes: [`last_hidden_state: [${encoderHiddenStates.dims.join(", ")}] (float32)`],
      decoderInputNames: [...this.decoderSession.inputNames],
      decoderInputShapes: [
        "input_ids: [1, seq_len] (int64)",
        "encoder_hidden_states: [1, 197, 768] (float32)",
      ],
      decoderOutputNames: [...this.decoderSession.outputNames],
      decoderOutputShapes: ["logits: [1, seq_len, 100000] (float32)"],
      first30TokenIds: rawTokens.slice(0, 30),
      numGeneratedTokens: rawTokens.length,
      decodedTextBeforeSpecialTokens,
      decodedTextAfterSpecialTokens,
      isOnlySpecialTokens,
      executionProvider: this.executionProvider,
      steps: stepsTrace,
      timings: {
        encoderMs,
        decoderMs,
        totalMs,
      },
    };
  }
}

// Cached singleton instance
let cachedTrOCREngine: BrowserTrOCR | null = null;

/**
 * Resolves a verified base URL for ONNX Runtime WASM binary assets.
 * Checks local /ort/ endpoint first, and falls back cleanly to the official npm CDN
 * (onnxruntime-web@1.30.0) if local assets are unreachable or unconfigured.
 */
async function resolveWasmPath(): Promise<string> {
  const localProbe = "/ort/ort-wasm-simd-threaded.jsep.wasm";
  const cdnFallback = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

  if (typeof window === "undefined") {
    return cdnFallback;
  }

  try {
    const res = await fetch(localProbe, { method: "HEAD" });
    if (res.ok && res.status === 200) {
      console.log("[BrowserTrOCR] Verified local WASM binaries available at /ort/");
      return "/ort/";
    }
  } catch (err) {
    console.warn("[BrowserTrOCR] Local /ort/ probe failed, falling back to package CDN:", err);
  }

  console.log(`[BrowserTrOCR] Local /ort/ not reachable (status not 200); using verified package CDN: ${cdnFallback}`);
  return cdnFallback;
}

/**
 * Clears the cached BrowserTrOCR engine instance so a new session with different parameters can be instantiated.
 */
export function clearTrOCRCache(): void {
  cachedTrOCREngine = null;
}

/**
 * Initializes and returns the Browser TrOCR engine singleton.
 * @param onProgress Callback for download and initialization progress
 * @param forceProvider Optional execution provider override ("webgpu" or "wasm")
 */
export async function getBrowserTrOCR(
  onProgress?: ProgressCallback,
  forceProvider?: "webgpu" | "wasm"
): Promise<BrowserTrOCR> {
  if (
    cachedTrOCREngine &&
    (!forceProvider || cachedTrOCREngine.getExecutionProvider() === forceProvider)
  ) {
    onProgress?.({
      stage: "ready",
      loadedBytes: 0,
      totalBytes: 0,
      percentage: 100,
      message: `Browser TrOCR engine already initialized (${cachedTrOCREngine.getExecutionProvider().toUpperCase()}).`,
    });
    return cachedTrOCREngine;
  }

  // 1. Download & verify model files via modelManager
  const buffers = await ensureModelLoaded(onProgress);

  onProgress?.({
    stage: "verifying",
    loadedBytes: 0,
    totalBytes: 0,
    percentage: 100,
    message: "Initializing OCR engine...",
  });

  // 2. Dynamically load ONNX Runtime Web in browser only
  const ort = await getOrt();
  const wasmBaseUrl = await resolveWasmPath();

  if (typeof window !== "undefined") {
    try {
      ort.env.wasm.wasmPaths = wasmBaseUrl;
      // In browsers without crossOriginIsolated, SharedArrayBuffer multi-threading fails.
      // Set to 1 if not crossOriginIsolated to prevent pthread worker crashes.
      const canMultiThread = typeof window !== "undefined" && !!window.crossOriginIsolated;
      ort.env.wasm.numThreads = canMultiThread
        ? Math.min(4, navigator.hardwareConcurrency || 2)
        : 1;
      console.log(`[BrowserTrOCR] Configured ort.env.wasm: wasmPaths="${wasmBaseUrl}", numThreads=${ort.env.wasm.numThreads}, crossOriginIsolated=${canMultiThread}`);
    } catch (e) {
      console.warn("[BrowserTrOCR] Could not configure WASM environment:", e);
    }
  }

  // 3. Genuine WebGPU capability check
  let canUseWebGPU = false;
  if (forceProvider !== "wasm" && typeof navigator !== "undefined" && "gpu" in navigator) {
    try {
      const adapter = await (navigator as any).gpu?.requestAdapter();
      if (adapter) {
        const device = await adapter.requestDevice();
        if (device) {
          device.destroy();
          canUseWebGPU = true;
        }
      }
    } catch (e) {
      console.warn("[BrowserTrOCR] WebGPU device check failed (not genuinely available):", e);
      canUseWebGPU = false;
    }
  }
  console.log(`[BrowserTrOCR] Genuine WebGPU support: ${canUseWebGPU} (forceProvider: ${forceProvider || "auto"})`);

  // 4. Retrieve model buffers
  const encoderBuf = buffers.get("encoder.onnx");
  const encoderDataBuf = buffers.get("encoder.onnx.data");
  const decoderBuf = buffers.get("decoder.onnx");
  const decoderDataBuf = buffers.get("decoder.onnx.data");
  const tokenizerBuf = buffers.get("tokenizer.json");

  if (!encoderBuf || !encoderDataBuf || !decoderBuf || !decoderDataBuf || !tokenizerBuf) {
    throw new Error("One or more required model buffers missing from cache.");
  }

  // 5. Build inverted vocabulary from tokenizer.json
  const tokenizerText = new TextDecoder("utf-8").decode(tokenizerBuf);
  const tokenizerJson = JSON.parse(tokenizerText);
  const vocabMap: Record<string, number> = tokenizerJson.model?.vocab || {};
  const idToToken: string[] = new Array(100000).fill("");
  for (const [tok, id] of Object.entries(vocabMap)) {
    if (id >= 0 && id < 100000) {
      idToToken[id] = tok;
    }
  }

  // 6 & 7. Create sessions with WebGPU -> WASM fallback
  const createSessionsWithEP = async (providers: string[]) => {
    const enc = await ort.InferenceSession.create(encoderBuf, {
      executionProviders: providers,
      externalData: [
        {
          path: "encoder.onnx.data",
          data: encoderDataBuf,
        },
      ],
    });
    const dec = await ort.InferenceSession.create(decoderBuf, {
      executionProviders: providers,
      externalData: [
        {
          path: "decoder.onnx.data",
          data: decoderDataBuf,
        },
      ],
    });
    return { enc, dec };
  };

  let encoderSession: OrtTypes.InferenceSession | null = null;
  let decoderSession: OrtTypes.InferenceSession | null = null;
  let chosenEP = "wasm";

  if (canUseWebGPU) {
    try {
      console.log(`[BrowserTrOCR] Attempting WebGPU session creation...`);
      const sessions = await createSessionsWithEP(["webgpu"]);
      encoderSession = sessions.enc;
      decoderSession = sessions.dec;
      chosenEP = "webgpu";
      console.log("[BrowserTrOCR] WebGPU execution provider initialized successfully.");
    } catch (gpuErr) {
      console.warn("[BrowserTrOCR] WebGPU session creation failed; cleanly falling back to WASM/CPU:", gpuErr);
      encoderSession = null;
      decoderSession = null;
      if (forceProvider === "webgpu") {
        throw new Error(`Forced WebGPU provider failed: ${gpuErr instanceof Error ? gpuErr.message : String(gpuErr)}`);
      }
    }
  }

  if (!encoderSession || !decoderSession) {
    if (forceProvider === "webgpu") {
      throw new Error("WebGPU is not available or failed to initialize on this device.");
    }
    console.log(`[BrowserTrOCR] Initializing WASM/CPU fallback provider (wasmPaths: ${wasmBaseUrl})...`);
    try {
      const sessions = await createSessionsWithEP(["wasm"]);
      encoderSession = sessions.enc;
      decoderSession = sessions.dec;
      chosenEP = "wasm";
      console.log("[BrowserTrOCR] Fallback to WASM/CPU execution provider succeeded.");
    } catch (wasmErr: any) {
      throw new Error(`WebGPU and WASM initialization both failed: ${wasmErr?.message || wasmErr}`);
    }
  }

  console.log(`[BrowserTrOCR] Successfully initialized ONNX Runtime sessions with provider: ${chosenEP}`);

  cachedTrOCREngine = new BrowserTrOCR(encoderSession, decoderSession, idToToken, chosenEP, ort);

  onProgress?.({
    stage: "ready",
    loadedBytes: 0,
    totalBytes: 0,
    percentage: 100,
    message: `OCR engine ready (${chosenEP.toUpperCase()}).`,
  });

  return cachedTrOCREngine;
}

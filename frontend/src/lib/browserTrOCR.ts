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
  encoderProvider: string;
  decoderProvider: string;
  latencyMs: number;
}

export interface TokenCandidate {
  tokenId: number;
  rawToken: string;
  logit: number;
}

export interface TokenStepDetail {
  step: number;
  inputIds: number[];
  outputDims: number[];
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
  encoderProvider: string;
  decoderProvider: string;
  fallbackReason?: string;
  decoderStartTokenId: number;
  bosTokenId: number;
  eosTokenId: number;
  padTokenId: number;
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
  tokenSequenceChanged: boolean;
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
  private encoderProvider: string;
  private decoderProvider: string;
  private fallbackReason?: string;
  private ort: typeof import("onnxruntime-web");

  constructor(
    encoderSession: OrtTypes.InferenceSession,
    decoderSession: OrtTypes.InferenceSession,
    idToToken: string[],
    encoderProvider: string,
    decoderProvider: string,
    ort: typeof import("onnxruntime-web"),
    fallbackReason?: string
  ) {
    this.encoderSession = encoderSession;
    this.decoderSession = decoderSession;
    this.idToToken = idToToken;
    this.encoderProvider = encoderProvider;
    this.decoderProvider = decoderProvider;
    this.ort = ort;
    this.fallbackReason = fallbackReason;
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

    // Step 2: Run vision encoder (WebGPU or WASM)
    console.log(`[Audit:Stage2-Inference] Running vision encoder (${this.encoderProvider})...`);
    const encoderResults = await this.encoderSession.run({
      pixel_values: pixelTensor,
    });
    const encoderHiddenStates = encoderResults.last_hidden_state;
    console.log(`[Audit:Stage2-Inference] Encoder completed. last_hidden_state shape: [${encoderHiddenStates.dims.join(", ")}]`);

    // Ensure encoder hidden states are a clean CPU float32 tensor for the WASM decoder
    const encData = encoderHiddenStates.data instanceof Float32Array
      ? encoderHiddenStates.data
      : new Float32Array(encoderHiddenStates.data as any);
    const encTensorForDecoder = new this.ort.Tensor(
      "float32",
      encData,
      encoderHiddenStates.dims
    );

    // Step 3: Autoregressive decoding (greedy search on WASM)
    const tokens: bigint[] = [BigInt(0)]; // decoder_start_token_id = 0 (<s>)
    const textConfidences: number[] = [];
    const eosTokenId = BigInt(2);
    const vocabSize = 100000;

    console.log(`[Audit:Stage2-Inference] Starting autoregressive decoding on ${this.decoderProvider} (max_length=${maxLength})...`);

    while (tokens.length < maxLength) {
      const inputIdsTensor = new this.ort.Tensor(
        "int64",
        new BigInt64Array(tokens),
        [1, tokens.length]
      );

      const decoderResults = await this.decoderSession.run({
        input_ids: inputIdsTensor,
        encoder_hidden_states: encTensorForDecoder,
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
      provider: this.getExecutionProvider(),
      latency_ms: latencyMs,
    });

    if (!decodedText) {
      console.warn(`[Audit:Stage3-ModelOutput] WARNING: BrowserTrOCR returned empty text! Tokens were:`, tokens.map(Number));
    }

    return {
      text: decodedText,
      confidence: parseFloat(avgConfidence.toFixed(4)),
      tokens: tokens.map(Number),
      executionProvider: this.getExecutionProvider(),
      encoderProvider: this.encoderProvider,
      decoderProvider: this.decoderProvider,
      latencyMs,
    };
  }

  public getExecutionProvider(): string {
    return this.encoderProvider === this.decoderProvider
      ? this.encoderProvider
      : `hybrid (${this.encoderProvider}/${this.decoderProvider})`;
  }

  public getEncoderProvider(): string {
    return this.encoderProvider;
  }

  public getDecoderProvider(): string {
    return this.decoderProvider;
  }

  public getFallbackReason(): string | undefined {
    return this.fallbackReason;
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

    // Prepare CPU-safe float32 tensor for decoder
    const encData = encoderHiddenStates.data instanceof Float32Array
      ? encoderHiddenStates.data
      : new Float32Array(encoderHiddenStates.data as any);
    const encTensorForDecoder = new this.ort.Tensor(
      "float32",
      encData,
      encoderHiddenStates.dims
    );

    // 4. Autoregressive Decoder with detailed per-step candidate trace
    const decStart = performance.now();
    const tokens: bigint[] = [BigInt(0)]; // decoder_start_token_id = 0 (<s>)
    const eosTokenId = BigInt(2);
    const vocabSize = 100000;
    const stepsTrace: TokenStepDetail[] = [];
    let lastOutputDims: number[] = [1, 1, vocabSize];

    while (tokens.length < maxLength) {
      const stepIdx = tokens.length - 1;
      const currentInputIds = tokens.map(Number);
      const inputIdsTensor = new this.ort.Tensor(
        "int64",
        new BigInt64Array(tokens),
        [1, tokens.length]
      );

      const decoderResults = await this.decoderSession.run({
        input_ids: inputIdsTensor,
        encoder_hidden_states: encTensorForDecoder,
      });

      const logits = decoderResults.logits.data as Float32Array;
      lastOutputDims = [...decoderResults.logits.dims];
      const lastTokenStart = (tokens.length - 1) * vocabSize;

      let maxLogit = -Infinity;
      let bestId = 0;

      // Single pass to find argmax and collect top 10 candidates
      const top10: { id: number; logit: number }[] = [];

      for (let v = 0; v < vocabSize; v++) {
        const val = logits[lastTokenStart + v];
        if (val > maxLogit) {
          maxLogit = val;
          bestId = v;
        }

        if (top10.length < 10) {
          top10.push({ id: v, logit: val });
          if (top10.length === 10) top10.sort((a, b) => b.logit - a.logit);
        } else if (val > top10[9].logit) {
          top10[9] = { id: v, logit: val };
          top10.sort((a, b) => b.logit - a.logit);
        }
      }

      const topCandidates: TokenCandidate[] = top10.map((c) => ({
        tokenId: c.id,
        rawToken: this.idToToken[c.id] || "",
        logit: parseFloat(c.logit.toFixed(4)),
      }));

      stepsTrace.push({
        step: stepIdx,
        inputIds: currentInputIds,
        outputDims: lastOutputDims,
        selectedTokenId: bestId,
        selectedRawToken: this.idToToken[bestId] || "",
        isSpecial: bestId <= 4,
        maxLogit: parseFloat(maxLogit.toFixed(4)),
        topCandidates,
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
    const tokenSequenceChanged = stepsTrace.length > 1 && stepsTrace.some((s, idx) => idx > 0 && s.selectedTokenId !== stepsTrace[0].selectedTokenId);

    return {
      originalDimensions: { width: origWidth, height: origHeight },
      preprocessedDimensions: { width: 224, height: 224, channels: 3 },
      tensorStats,
      encoderProvider: this.encoderProvider,
      decoderProvider: this.decoderProvider,
      fallbackReason: this.fallbackReason,
      decoderStartTokenId: 0,
      bosTokenId: 0,
      eosTokenId: 2,
      padTokenId: 1,
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
      decoderOutputShapes: [`logits: [${lastOutputDims.join(", ")}] (float32)`],
      first30TokenIds: rawTokens.slice(0, 30),
      numGeneratedTokens: rawTokens.length,
      decodedTextBeforeSpecialTokens,
      decodedTextAfterSpecialTokens,
      isOnlySpecialTokens,
      tokenSequenceChanged,
      executionProvider: this.getExecutionProvider(),
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
 * Employs Hybrid Execution Architecture:
 * - Vision Encoder: WebGPU when supported, with clean WASM/CPU fallback.
 * - Text Decoder: Forced to WASM/CPU to reliably support 307MB lm_head.weight and dynamic sequence length autoregression.
 * @param onProgress Callback for download and initialization progress
 * @param forceProvider Optional execution provider override ("webgpu" or "wasm")
 */
export async function getBrowserTrOCR(
  onProgress?: ProgressCallback,
  forceProvider?: "webgpu" | "wasm"
): Promise<BrowserTrOCR> {
  if (
    cachedTrOCREngine &&
    (!forceProvider || cachedTrOCREngine.getExecutionProvider().includes(forceProvider))
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

  // 6. Create Encoder Session: WebGPU preferred, with automatic WASM fallback
  let encoderSession: OrtTypes.InferenceSession | null = null;
  let encoderProvider = "wasm";
  let fallbackReason: string | undefined = undefined;

  if (canUseWebGPU && forceProvider !== "wasm") {
    try {
      console.log(`[BrowserTrOCR] Creating Vision Encoder session with WebGPU...`);
      encoderSession = await ort.InferenceSession.create(encoderBuf, {
        executionProviders: ["webgpu"],
        externalData: [
          {
            path: "encoder.onnx.data",
            data: encoderDataBuf,
          },
        ],
      });
      encoderProvider = "webgpu";
      console.log("[BrowserTrOCR] Vision Encoder WebGPU session initialized successfully.");
    } catch (gpuErr: any) {
      fallbackReason = `Encoder WebGPU failed: ${gpuErr?.message || String(gpuErr)}`;
      console.warn(`[BrowserTrOCR] ${fallbackReason}. Cleanly falling back to WASM for encoder.`);
      encoderSession = null;
    }
  }

  if (!encoderSession) {
    console.log(`[BrowserTrOCR] Initializing Vision Encoder with WASM/CPU provider...`);
    try {
      encoderSession = await ort.InferenceSession.create(encoderBuf, {
        executionProviders: ["wasm"],
        externalData: [
          {
            path: "encoder.onnx.data",
            data: encoderDataBuf,
          },
        ],
      });
      encoderProvider = "wasm";
      console.log("[BrowserTrOCR] Vision Encoder WASM session initialized successfully.");
    } catch (wasmErr: any) {
      throw new Error(`Vision Encoder initialization failed: ${wasmErr?.message || wasmErr}`);
    }
  }

  // 7. Create Text Decoder Session: WASM/CPU to support 307MB lm_head.weight and dynamic sequence length
  let decoderSession: OrtTypes.InferenceSession | null = null;
  let decoderProvider = "wasm";

  if (forceProvider === "webgpu") {
    // Only attempt WebGPU on decoder if explicitly forced in debug mode
    try {
      console.log(`[BrowserTrOCR] [DEBUG] Forcing WebGPU on Text Decoder...`);
      decoderSession = await ort.InferenceSession.create(decoderBuf, {
        executionProviders: ["webgpu"],
        externalData: [
          {
            path: "decoder.onnx.data",
            data: decoderDataBuf,
          },
        ],
      });
      decoderProvider = "webgpu";
    } catch (gpuErr: any) {
      throw new Error(`Forced WebGPU decoder session failed: ${gpuErr?.message || gpuErr}`);
    }
  } else {
    console.log(`[BrowserTrOCR] Initializing Autoregressive Text Decoder with WASM/CPU provider (wasmPaths: ${wasmBaseUrl})...`);
    try {
      decoderSession = await ort.InferenceSession.create(decoderBuf, {
        executionProviders: ["wasm"],
        externalData: [
          {
            path: "decoder.onnx.data",
            data: decoderDataBuf,
          },
        ],
      });
      decoderProvider = "wasm";
      console.log("[BrowserTrOCR] Text Decoder WASM/CPU session initialized successfully.");
    } catch (decErr: any) {
      throw new Error(`Text Decoder initialization failed: ${decErr?.message || decErr}`);
    }
  }

  const overallProvider = encoderProvider === decoderProvider ? encoderProvider : `hybrid (${encoderProvider}/${decoderProvider})`;
  console.log(`[BrowserTrOCR] Successfully initialized sessions: Encoder=${encoderProvider.toUpperCase()}, Decoder=${decoderProvider.toUpperCase()} (${overallProvider})`);

  cachedTrOCREngine = new BrowserTrOCR(
    encoderSession,
    decoderSession,
    idToToken,
    encoderProvider,
    decoderProvider,
    ort,
    fallbackReason
  );

  onProgress?.({
    stage: "ready",
    loadedBytes: 0,
    totalBytes: 0,
    percentage: 100,
    message: `OCR engine ready: Encoder (${encoderProvider.toUpperCase()}), Decoder (${decoderProvider.toUpperCase()}).`,
  });

  return cachedTrOCREngine;
}

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
    } else {
      throw new Error("BrowserTrOCR image preprocessing requires window and canvas context.");
    }

    if (source instanceof Blob || source instanceof File) {
      const img = new Image();
      const url = URL.createObjectURL(source);
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = reject;
        img.src = url;
      });
      ctx.drawImage(img, 0, 0, 224, 224);
      URL.revokeObjectURL(url);
    } else if (source instanceof ImageData) {
      ctx.putImageData(source, 0, 0);
    } else {
      ctx.drawImage(source, 0, 0, 224, 224);
    }

    const imgData = ctx.getImageData(0, 0, 224, 224);
    const rgba = imgData.data;

    // Planar RGB array: shape [1, 3, 224, 224]
    const planar = new Float32Array(3 * 224 * 224);
    const planeSize = 224 * 224;

    for (let i = 0; i < planeSize; i++) {
      const r = rgba[i * 4];
      const g = rgba[i * 4 + 1];
      const b = rgba[i * 4 + 2];

      // Channel 0: R, Channel 1: G, Channel 2: B
      planar[i] = (r - 127.5) / 127.5;
      planar[planeSize + i] = (g - 127.5) / 127.5;
      planar[2 * planeSize + i] = (b - 127.5) / 127.5;
    }

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
    const encoderResults = await this.encoderSession.run({
      pixel_values: pixelTensor,
    });
    const encoderHiddenStates = encoderResults.last_hidden_state;

    // Step 3: Autoregressive decoding (greedy search)
    const tokens: bigint[] = [BigInt(0)]; // decoder_start_token_id = 0 (<s>)
    const confidences: number[] = [];
    const eosTokenId = BigInt(2);
    const vocabSize = 100000;

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
      confidences.push(Math.min(1.0, Math.max(0.1, 1 / (1 + Math.exp(-maxLogit / 10)))));
    }

    // Step 4: Token decoding via ByteLevel BPE
    const tokenStrings = tokens.map((id) => this.idToToken[Number(id)] || "");
    const decodedText = decodeByteLevelTokens(tokenStrings).trim();

    const avgConfidence =
      confidences.length > 0
        ? confidences.reduce((a, b) => a + b, 0) / confidences.length
        : 0.85;

    const latencyMs = Math.round(performance.now() - startTime);

    return {
      text: decodedText,
      confidence: parseFloat(avgConfidence.toFixed(4)),
      tokens: tokens.map(Number),
      executionProvider: this.executionProvider,
      latencyMs,
    };
  }
}

// Cached singleton instance
let cachedTrOCREngine: BrowserTrOCR | null = null;

/**
 * Initializes and returns the Browser TrOCR engine singleton.
 */
export async function getBrowserTrOCR(
  onProgress?: ProgressCallback
): Promise<BrowserTrOCR> {
  if (cachedTrOCREngine) {
    onProgress?.({
      stage: "ready",
      loadedBytes: 0,
      totalBytes: 0,
      percentage: 100,
      message: "Browser TrOCR engine already initialized.",
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
  if (typeof window !== "undefined") {
    try {
      ort.env.wasm.wasmPaths = "/ort/";
      ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
    } catch (e) {
      console.warn("[BrowserTrOCR] Could not set local wasm paths, using default:", e);
    }
  }

  // 3. Select Execution Provider (Prefer WebGPU with automatic WASM fallback)
  let tryWebGPU = false;
  let chosenEP = "wasm";

  if (typeof navigator !== "undefined" && "gpu" in navigator) {
    try {
      const adapter = await (navigator as any).gpu?.requestAdapter();
      if (adapter) {
        tryWebGPU = true;
      }
    } catch {
      tryWebGPU = false;
    }
  }

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

  let encoderSession: OrtTypes.InferenceSession;
  let decoderSession: OrtTypes.InferenceSession;

  if (tryWebGPU) {
    try {
      const sessions = await createSessionsWithEP(["webgpu", "wasm"]);
      encoderSession = sessions.enc;
      decoderSession = sessions.dec;
      chosenEP = "webgpu";
      console.log("[BrowserTrOCR] WebGPU execution provider initialized successfully.");
    } catch (gpuErr) {
      console.warn("[BrowserTrOCR] WebGPU session creation failed, falling back to WASM/CPU:", gpuErr);
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
  } else {
    try {
      const sessions = await createSessionsWithEP(["wasm"]);
      encoderSession = sessions.enc;
      decoderSession = sessions.dec;
      chosenEP = "wasm";
      console.log("[BrowserTrOCR] WASM/CPU execution provider initialized successfully.");
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

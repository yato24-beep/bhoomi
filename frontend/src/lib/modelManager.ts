/**
 * Automatic Browser Model Manager for IIT Bombay Indic-TrOCR Kannada Checkpoint-12000.
 *
 * Implements:
 * - Versioned manifest fetching (/model_manifest.json)
 * - Cache inspection & version invalidation via CacheStorage
 * - File size and SHA-256 cryptographic verification
 * - Deletion of corrupted or incomplete files
 * - Streaming download with real progress reporting
 * - Concurrency lock preventing duplicate simultaneous downloads
 * - Clean quota & network failure handling
 */

export interface ModelFileMeta {
  filename: string;
  size: number;
  sha256: string;
  url: string;
  fallback_url?: string;
  required: boolean;
}

export interface ModelManifest {
  manifest_version: string;
  model_name: string;
  model_version: string;
  format: string;
  huggingface_repo: string;
  total_size_bytes: number;
  total_size_mb: number;
  signatures: {
    encoder: any;
    decoder: any;
  };
  generation: {
    bos_token_id: number;
    decoder_start_token_id: number;
    eos_token_id: number;
    pad_token_id: number;
    max_length: number;
  };
  files: Record<string, ModelFileMeta>;
}

export interface DownloadProgress {
  stage: "checking" | "downloading" | "verifying" | "ready" | "error";
  currentFile?: string;
  loadedBytes: number;
  totalBytes: number;
  percentage: number;
  message: string;
}

export type ProgressCallback = (progress: DownloadProgress) => void;

let activeDownloadPromise: Promise<Map<string, ArrayBuffer>> | null = null;

/**
 * Computes SHA-256 hex string for an ArrayBuffer using Web Cryptography API.
 */
export async function computeSHA256(buffer: ArrayBuffer): Promise<string> {
  const hashBuffer = await crypto.subtle.digest("SHA-256", buffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * Fetches the versioned model manifest.
 */
export async function fetchModelManifest(): Promise<ModelManifest> {
  const res = await fetch("/model_manifest.json", { cache: "no-cache" });
  if (!res.ok) {
    throw new Error(`Failed to load model manifest (${res.status} ${res.statusText})`);
  }
  return res.json();
}

/**
 * Returns the Cache name for a given model version.
 */
function getCacheName(version: string): string {
  return `bhoomi-trocr-kannada-v${version}`;
}

/**
 * Purges old model versions from CacheStorage if model version changes.
 */
async function purgeOutdatedCaches(currentVersion: string): Promise<void> {
  if (typeof window === "undefined" || !("caches" in window)) return;
  const currentCacheName = getCacheName(currentVersion);
  const keys = await caches.keys();
  for (const key of keys) {
    if (key.startsWith("bhoomi-trocr-kannada-v") && key !== currentCacheName) {
      console.log(`[ModelManager] Purging outdated cache: ${key}`);
      await caches.delete(key);
    }
  }
}

/**
 * Checks cache status for all required files in the manifest.
 */
export async function checkModelCacheStatus(
  manifest: ModelManifest
): Promise<{ isComplete: boolean; cachedFiles: Set<string>; missingFiles: string[] }> {
  if (typeof window === "undefined" || !("caches" in window)) {
    return { isComplete: false, cachedFiles: new Set(), missingFiles: Object.keys(manifest.files) };
  }

  await purgeOutdatedCaches(manifest.model_version);
  const cacheName = getCacheName(manifest.model_version);
  const cache = await caches.open(cacheName);

  const cachedFiles = new Set<string>();
  const missingFiles: string[] = [];

  for (const [filename, meta] of Object.entries(manifest.files)) {
    const key = `/__trocr_models__/${filename}`;
    const response = await cache.match(key);
    if (!response) {
      missingFiles.push(filename);
      continue;
    }

    const blob = await response.blob();
    if (blob.size !== meta.size) {
      console.warn(`[ModelManager] Corrupted file size for ${filename}: expected ${meta.size}, got ${blob.size}. Deleting.`);
      await cache.delete(key);
      missingFiles.push(filename);
      continue;
    }

    cachedFiles.add(filename);
  }

  return {
    isComplete: missingFiles.length === 0,
    cachedFiles,
    missingFiles,
  };
}

/**
 * Downloads a single file with streaming progress and stores it in CacheStorage.
 */
async function downloadAndCacheFile(
  cache: Cache,
  meta: ModelFileMeta,
  onByteDelta: (bytes: number) => void
): Promise<ArrayBuffer> {
  const fileKey = `/__trocr_models__/${meta.filename}`;
  const urlsToTry = [meta.url];
  if (meta.fallback_url && meta.fallback_url !== meta.url) {
    urlsToTry.push(meta.fallback_url);
  }

  let lastError: any = null;

  for (const url of urlsToTry) {
    try {
      const res = await fetch(url, { mode: "cors" });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }

      if (!res.body) {
        const buffer = await res.arrayBuffer();
        onByteDelta(buffer.byteLength);
        await cache.put(fileKey, new Response(buffer));
        return buffer;
      }

      const reader = res.body.getReader();
      const chunks: Uint8Array[] = [];
      let receivedBytes = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value) {
          chunks.push(value);
          receivedBytes += value.byteLength;
          onByteDelta(value.byteLength);
        }
      }

      // Concatenate chunks into single buffer
      const fullBuffer = new Uint8Array(receivedBytes);
      let offset = 0;
      for (const chunk of chunks) {
        fullBuffer.set(chunk, offset);
        offset += chunk.byteLength;
      }

      const arrayBuffer = fullBuffer.buffer;

      // Verify size
      if (arrayBuffer.byteLength !== meta.size) {
        throw new Error(
          `Corrupted download for ${meta.filename}: Expected ${meta.size} bytes, got ${arrayBuffer.byteLength}`
        );
      }

      // Verify SHA-256
      if (meta.sha256) {
        const hash = await computeSHA256(arrayBuffer);
        if (hash !== meta.sha256) {
          throw new Error(
            `SHA-256 checksum mismatch for ${meta.filename}! Expected ${meta.sha256}, calculated ${hash}`
          );
        }
      }

      // Store in CacheStorage
      try {
        await cache.put(fileKey, new Response(arrayBuffer));
      } catch (storageErr: any) {
        if (storageErr.name === "QuotaExceededError") {
          throw new Error(
            "Browser storage quota exceeded. Please free up browser disk space or enable persistent storage permissions."
          );
        }
        throw storageErr;
      }

      return arrayBuffer;
    } catch (err) {
      console.warn(`[ModelManager] Download attempt failed for ${url}:`, err);
      lastError = err;
    }
  }

  throw new Error(`Failed to download ${meta.filename} from all available sources: ${lastError?.message || lastError}`);
}

/**
 * Ensures all model files are available in cache and returns them as ArrayBuffers.
 * Safe against duplicate concurrent downloads.
 */
export async function ensureModelLoaded(
  onProgress?: ProgressCallback
): Promise<Map<string, ArrayBuffer>> {
  if (activeDownloadPromise) {
    return activeDownloadPromise;
  }

  activeDownloadPromise = (async () => {
    try {
      onProgress?.({
        stage: "checking",
        loadedBytes: 0,
        totalBytes: 0,
        percentage: 0,
        message: "Fetching model manifest...",
      });

      const manifest = await fetchModelManifest();
      const cacheName = getCacheName(manifest.model_version);
      const cache = await caches.open(cacheName);

      const status = await checkModelCacheStatus(manifest);
      const totalBytes = manifest.total_size_bytes;
      let loadedBytes = 0;

      const loadedBuffers = new Map<string, ArrayBuffer>();

      // Load already cached valid files
      for (const filename of Array.from(status.cachedFiles)) {
        const meta = manifest.files[filename];
        const key = `/__trocr_models__/${filename}`;
        const match = await cache.match(key);
        if (match) {
          const buf = await match.arrayBuffer();
          loadedBuffers.set(filename, buf);
          loadedBytes += meta.size;
        }
      }

      if (status.missingFiles.length === 0) {
        onProgress?.({
          stage: "ready",
          loadedBytes: totalBytes,
          totalBytes,
          percentage: 100,
          message: "IITB Indic-TrOCR model loaded from local cache.",
        });
        return loadedBuffers;
      }

      // Download missing files
      onProgress?.({
        stage: "downloading",
        loadedBytes,
        totalBytes,
        percentage: Math.round((loadedBytes / totalBytes) * 100),
        message: "Downloading OCR model...",
      });

      for (const filename of status.missingFiles) {
        const meta = manifest.files[filename];
        onProgress?.({
          stage: "downloading",
          currentFile: filename,
          loadedBytes,
          totalBytes,
          percentage: Math.round((loadedBytes / totalBytes) * 100),
          message: `Downloading OCR model (${filename})...`,
        });

        const buf = await downloadAndCacheFile(cache, meta, (delta) => {
          loadedBytes += delta;
          onProgress?.({
            stage: "downloading",
            currentFile: filename,
            loadedBytes,
            totalBytes,
            percentage: Math.min(99, Math.round((loadedBytes / totalBytes) * 100)),
            message: `Downloading OCR model (${filename})...`,
          });
        });

        loadedBuffers.set(filename, buf);
      }

      onProgress?.({
        stage: "ready",
        loadedBytes: totalBytes,
        totalBytes,
        percentage: 100,
        message: "OCR model downloaded and verified.",
      });

      return loadedBuffers;
    } catch (err: any) {
      onProgress?.({
        stage: "error",
        loadedBytes: 0,
        totalBytes: 0,
        percentage: 0,
        message: `Model download failed: ${err?.message || err}`,
      });
      throw new Error(`Model download failed: ${err?.message || err}`);
    } finally {
      activeDownloadPromise = null;
    }
  })();

  return activeDownloadPromise;
}

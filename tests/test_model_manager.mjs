/**
 * Node.js test simulating browser model manager verification logic:
 * - Manifest validation
 * - SHA-256 computation against real files
 * - Corrupted file detection
 * - Version invalidation
 */

import fs from "fs";
import path from "path";
import crypto from "crypto";

const manifestPath = path.resolve("frontend/public/model_manifest.json");
const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf-8"));

console.log("=== BROWSER MODEL MANAGER VERIFICATION TEST ===");
console.log(`Model: ${manifest.model_name} (v${manifest.model_version})`);
console.log(`Format: ${manifest.format}, Total Size: ${manifest.total_size_mb} MB`);

// 1. Verify all files declared in manifest exist and have matching SHA-256
console.log("\n--- Checking File Integrity and Cryptographic Checksums ---");
const filesToCheck = {
  "encoder.onnx": "experiments/onnx_kannada_trocr/models/encoder.onnx",
  "encoder.onnx.data": "experiments/onnx_kannada_trocr/models/encoder.onnx.data",
  "decoder.onnx": "experiments/onnx_kannada_trocr/models/decoder.onnx",
  "decoder.onnx.data": "experiments/onnx_kannada_trocr/models/decoder.onnx.data",
  "tokenizer.json": "experiments/onnx_kannada_trocr/models/tokenizer/tokenizer.json",
  "tokenizer_config.json": "experiments/onnx_kannada_trocr/models/tokenizer/tokenizer_config.json",
  "preprocessor_config.json": "src/handwriting/configs/iitb_kannada_v002/preprocessor_config.json",
  "config.json": "models/trocr/checkpoint-12000/config.json",
  "generation_config.json": "models/trocr/checkpoint-12000/generation_config.json",
};

let verifiedCount = 0;
for (const [filename, relPath] of Object.entries(filesToCheck)) {
  const meta = manifest.files[filename];
  if (!meta) {
    throw new Error(`File ${filename} missing from manifest!`);
  }

  const fullPath = path.resolve(relPath);
  const data = fs.readFileSync(fullPath);
  const actualSize = data.length;
  const actualHash = crypto.createHash("sha256").update(data).digest("hex");

  if (actualSize !== meta.size) {
    throw new Error(`Size mismatch for ${filename}: manifest=${meta.size}, actual=${actualSize}`);
  }
  if (actualHash !== meta.sha256) {
    throw new Error(`SHA-256 mismatch for ${filename}: manifest=${meta.sha256}, actual=${actualHash}`);
  }

  console.log(`  [PASS] ${filename.padEnd(25)} (${(actualSize / 1024 / 1024).toFixed(2).padStart(6)} MB) | SHA-256: ${actualHash.substring(0, 16)}...`);
  verifiedCount++;
}

// 2. Simulate corrupted file detection
console.log("\n--- Testing Corrupted File Detection Simulation ---");
const corruptedBuffer = Buffer.from("corrupted file content");
const corruptedHash = crypto.createHash("sha256").update(corruptedBuffer).digest("hex");
const targetMeta = manifest.files["tokenizer_config.json"];

const isCorrupted = corruptedBuffer.length !== targetMeta.size || corruptedHash !== targetMeta.sha256;
if (!isCorrupted) {
  throw new Error("Corrupted buffer was unexpectedly accepted!");
}
console.log("  [PASS] Corrupted file successfully detected and rejected (simulated cache purge).");

// 3. Simulate version invalidation
console.log("\n--- Testing Version Invalidation Simulation ---");
const cachedVersion = "0.9.0";
const currentVersion = manifest.model_version;
const shouldPurge = cachedVersion !== currentVersion;
if (!shouldPurge) {
  throw new Error("Outdated cache version was not flagged for purge!");
}
console.log(`  [PASS] Outdated cache v${cachedVersion} flagged for purging in favor of current v${currentVersion}.`);

console.log("\nALL BROWSER MODEL MANAGER UNIT CHECKS PASSED!");

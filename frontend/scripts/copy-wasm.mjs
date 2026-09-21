import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const srcDir = path.resolve(__dirname, "../node_modules/onnxruntime-web/dist");
const destDir = path.resolve(__dirname, "../public/ort");

if (!fs.existsSync(destDir)) {
  fs.mkdirSync(destDir, { recursive: true });
}

if (fs.existsSync(srcDir)) {
  const files = fs.readdirSync(srcDir);
  let copiedCount = 0;
  for (const file of files) {
    if (file.startsWith("ort-wasm")) {
      const srcFile = path.join(srcDir, file);
      const destFile = path.join(destDir, file);
      fs.copyFileSync(srcFile, destFile);
      copiedCount++;
    }
  }
  console.log(`[copy-wasm] Copied ${copiedCount} ONNX Runtime WASM assets to public/ort/`);
} else {
  console.warn(`[copy-wasm] Source directory ${srcDir} does not exist. Skipping WASM copy.`);
}

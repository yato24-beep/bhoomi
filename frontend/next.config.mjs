import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  webpack: (config, { isServer }) => {
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      path: false,
      crypto: false,
    };
    if (isServer) {
      config.resolve.alias = {
        ...config.resolve.alias,
        "onnxruntime-web": false,
        "onnxruntime-web/webgpu": false,
      };
    } else {
      config.resolve.alias = {
        ...config.resolve.alias,
        "onnxruntime-web": path.resolve(__dirname, "node_modules/onnxruntime-web/dist/ort.all.min.js"),
        "onnxruntime-web/webgpu": path.resolve(__dirname, "node_modules/onnxruntime-web/dist/ort.all.min.js"),
      };
    }
    return config;
  },
};

export default nextConfig;

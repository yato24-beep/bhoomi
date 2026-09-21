"use client";

import React, { useState, useRef, useEffect } from "react";
import Link from "next/link";
import {
  Cpu,
  Upload,
  Play,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  FileImage,
  Layers,
  ArrowRight,
  Zap,
  Info,
  ChevronRight,
} from "lucide-react";
import {
  getBrowserTrOCR,
  clearTrOCRCache,
  OCRDiagnosticReport,
} from "@/lib/browserTrOCR";
import { DownloadProgress } from "@/lib/modelManager";

export default function OCRDiagnosticPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [sampleLoaded, setSampleLoaded] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progressMsg, setProgressMsg] = useState<string>("");
  const [progressPct, setProgressPct] = useState<number>(0);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [report, setReport] = useState<OCRDiagnosticReport | null>(null);
  const [forcedProvider, setForcedProvider] = useState<"auto" | "webgpu" | "wasm">("auto");
  const [maxLength, setMaxLength] = useState<number>(32);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const preprocessedCanvasRef = useRef<HTMLCanvasElement>(null);

  // Clean up object URLs
  useEffect(() => {
    return () => {
      if (previewUrl && previewUrl.startsWith("blob:")) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  // Handle local file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (previewUrl && previewUrl.startsWith("blob:")) {
        URL.revokeObjectURL(previewUrl);
      }
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setSampleLoaded(false);
      setReport(null);
      setErrorMsg(null);
    }
  };

  // Load sample Kannada crop
  const handleLoadSample = async () => {
    try {
      setErrorMsg(null);
      setReport(null);
      const res = await fetch("/sample_kannada_crop.png");
      if (!res.ok) {
        throw new Error("Could not load /sample_kannada_crop.png");
      }
      const blob = await res.blob();
      const file = new File([blob], "sample_kannada_crop.png", { type: "image/png" });
      setSelectedFile(file);
      setPreviewUrl("/sample_kannada_crop.png");
      setSampleLoaded(true);
    } catch (err: any) {
      setErrorMsg(`Failed to load sample image: ${err.message}`);
    }
  };

  // Draw 224x224 preprocessed image onto canvas for visual inspection
  useEffect(() => {
    if (!previewUrl || !preprocessedCanvasRef.current) return;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const canvas = preprocessedCanvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, 224, 224);
      ctx.drawImage(img, 0, 0, 224, 224);
    };
    img.src = previewUrl;
  }, [previewUrl, report]);

  // Run the in-browser diagnostic
  const handleRunDiagnostic = async () => {
    if (!selectedFile && !previewUrl) {
      setErrorMsg("Please upload an image or load the sample Kannada crop first.");
      return;
    }

    setIsProcessing(true);
    setErrorMsg(null);
    setReport(null);
    setProgressMsg("Starting diagnostic...");
    setProgressPct(5);

    try {
      const providerArg = forcedProvider === "auto" ? undefined : forcedProvider;
      
      const engine = await getBrowserTrOCR((progress: DownloadProgress) => {
        setProgressMsg(progress.message);
        setProgressPct(progress.percentage);
      }, providerArg);

      setProgressMsg("Running vision encoder and autoregressive decoder...");
      setProgressPct(90);

      // Pass the uploaded file or image directly
      const inputSource = selectedFile || (await (async () => {
        const resp = await fetch(previewUrl!);
        return await resp.blob();
      })());

      const diagReport = await engine.diagnose(inputSource, maxLength);
      setReport(diagReport);
      setProgressMsg("Diagnostic complete.");
      setProgressPct(100);
    } catch (err: any) {
      console.error("[OCR Diagnostics Error]", err);
      setErrorMsg(err.message || String(err));
    } finally {
      setIsProcessing(false);
    }
  };

  // Reset engine cache to allow re-initializing with different execution providers
  const handleResetEngine = () => {
    clearTrOCRCache();
    setReport(null);
    setProgressMsg("Engine cache cleared. Next run will initialize fresh sessions.");
    setProgressPct(0);
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 pb-20">
      {/* Header */}
      <div className="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-amber-500/20 text-amber-400 border border-amber-500/30 rounded-lg">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-bold text-white tracking-tight">
                  Browser-Only OCR Diagnostic
                </h1>
                <span className="px-2 py-0.5 text-[11px] font-mono font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/40 rounded">
                  ISOLATED DEBUG MODE
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Zero backend calls &bull; Zero Gemini &bull; Zero DB saves &bull; Real ONNX runtime tokens only
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Link
              href="/"
              className="text-xs text-slate-400 hover:text-slate-200 px-3 py-1.5 rounded-md hover:bg-slate-800 transition-colors"
            >
              Back to Dashboard
            </Link>
          </div>
        </div>
      </div>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-6 space-y-6">
        {/* Execution Provider & Control Toolbar */}
        <div className="p-4 bg-slate-800/80 border border-slate-700 rounded-xl flex flex-wrap items-center justify-between gap-4 shadow-lg">
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
              Execution Provider:
            </span>
            <div className="inline-flex rounded-lg bg-slate-900 p-1 border border-slate-700">
              {(["auto", "webgpu", "wasm"] as const).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => {
                    setForcedProvider(mode);
                    clearTrOCRCache();
                  }}
                  className={`px-3 py-1 text-xs font-mono font-medium rounded-md transition-all ${
                    forcedProvider === mode
                      ? "bg-amber-500 text-slate-950 font-bold shadow-sm"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  {mode === "auto" ? "Auto (WebGPU > WASM)" : mode.toUpperCase()}
                </button>
              ))}
            </div>

            <button
              type="button"
              onClick={handleResetEngine}
              title="Clear cached ONNX sessions to force reload"
              className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs text-slate-300 bg-slate-700/60 hover:bg-slate-700 rounded-md border border-slate-600 transition-colors"
            >
              <RefreshCw className="w-3 h-3" />
              Reset Engine Cache
            </button>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-xs text-slate-300 flex items-center gap-2">
              <span>Max Tokens:</span>
              <input
                type="number"
                min={8}
                max={64}
                value={maxLength}
                onChange={(e) => setMaxLength(Math.max(4, Math.min(64, parseInt(e.target.value) || 32)))}
                className="w-16 px-2 py-1 text-xs bg-slate-900 border border-slate-700 rounded text-amber-300 font-mono text-center focus:outline-none focus:border-amber-500"
              />
            </label>
          </div>
        </div>

        {/* Input & Upload Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Upload card */}
          <div className="lg:col-span-1 p-5 bg-slate-800/60 border border-slate-700 rounded-xl space-y-4">
            <h2 className="text-sm font-bold text-white flex items-center gap-2">
              <Upload className="w-4 h-4 text-amber-400" />
              1. Input Image
            </h2>

            {/* Drop Zone */}
            <div
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-slate-600 hover:border-amber-500/60 bg-slate-900/50 hover:bg-slate-900/80 rounded-xl p-6 text-center cursor-pointer transition-all flex flex-col items-center justify-center min-h-[160px]"
            >
              <FileImage className="w-8 h-8 text-slate-400 mb-2" />
              <p className="text-xs font-semibold text-slate-200">
                Click to upload handwritten Kannada image
              </p>
              <p className="text-[11px] text-slate-500 mt-1">PNG, JPG, WEBP, or TIFF</p>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
              />
            </div>

            <div className="flex items-center gap-2">
              <span className="text-[11px] text-slate-400">or</span>
              <button
                type="button"
                onClick={handleLoadSample}
                className="w-full text-xs font-semibold py-2 px-3 rounded-lg bg-slate-700 hover:bg-slate-600 text-amber-300 border border-slate-600 transition-colors text-center"
              >
                Use Built-in Sample Kannada Crop
              </button>
            </div>

            {/* Run Diagnostic Button */}
            <button
              type="button"
              disabled={isProcessing || (!selectedFile && !previewUrl)}
              onClick={handleRunDiagnostic}
              className={`w-full py-3 px-4 rounded-xl text-xs font-bold uppercase tracking-wider flex items-center justify-center gap-2 transition-all shadow-lg ${
                isProcessing || (!selectedFile && !previewUrl)
                  ? "bg-slate-700 text-slate-400 cursor-not-allowed"
                  : "bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-400 hover:to-orange-400 text-slate-950 shadow-amber-500/20"
              }`}
            >
              {isProcessing ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Running Diagnostic...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  Run Browser ONNX Diagnostic
                </>
              )}
            </button>

            {/* Progress indicator */}
            {isProcessing && (
              <div className="space-y-2 p-3 bg-slate-900/80 border border-slate-700 rounded-lg">
                <div className="flex justify-between text-[11px] text-slate-400 font-mono">
                  <span className="truncate pr-2">{progressMsg}</span>
                  <span>{progressPct}%</span>
                </div>
                <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                  <div
                    className="bg-amber-500 h-full transition-all duration-200"
                    style={{ width: `${progressPct}%` }}
                  />
                </div>
              </div>
            )}

            {/* Error Message */}
            {errorMsg && (
              <div className="p-3 bg-rose-950/60 border border-rose-800/80 rounded-lg text-rose-300 text-xs flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5 text-rose-400" />
                <div className="space-y-1">
                  <p className="font-bold">Execution Error</p>
                  <p className="text-[11px] font-mono break-all">{errorMsg}</p>
                </div>
              </div>
            )}
          </div>

          {/* Previews Card */}
          <div className="lg:col-span-2 p-5 bg-slate-800/60 border border-slate-700 rounded-xl space-y-4">
            <h2 className="text-sm font-bold text-white flex items-center gap-2">
              <Layers className="w-4 h-4 text-amber-400" />
              2. Preprocessing & Tensor Visualizer
            </h2>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* Original Preview */}
              <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-3 space-y-2 flex flex-col items-center justify-center min-h-[260px]">
                <span className="text-[11px] font-mono text-slate-400">
                  Uploaded Input Image {report ? `(${report.originalDimensions.width}x${report.originalDimensions.height})` : ""}
                </span>
                {previewUrl ? (
                  <div className="max-h-[220px] max-w-full overflow-hidden flex items-center justify-center">
                    <img
                      src={previewUrl}
                      alt="Uploaded image"
                      className="max-h-[200px] object-contain rounded border border-slate-700"
                    />
                  </div>
                ) : (
                  <p className="text-xs text-slate-600 italic">No image selected</p>
                )}
              </div>

              {/* Preprocessed 224x224 Canvas */}
              <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-3 space-y-2 flex flex-col items-center justify-center min-h-[260px]">
                <span className="text-[11px] font-mono text-slate-400">
                  Resized & Padded (224x224 ViT Input)
                </span>
                <canvas
                  ref={preprocessedCanvasRef}
                  width={224}
                  height={224}
                  className="w-[200px] h-[200px] rounded border border-slate-700 bg-white object-contain"
                />
                {report && (
                  <div className="text-[10px] font-mono text-slate-400 text-center">
                    Tensor min: <span className="text-amber-300">{report.tensorStats.min}</span> &bull; max:{" "}
                    <span className="text-amber-300">{report.tensorStats.max}</span> &bull; mean:{" "}
                    <span className="text-amber-300">{report.tensorStats.mean}</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Diagnostic Results Section */}
        {report && (
          <div className="space-y-6">
            {/* Top Stat Summary Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {/* Selected Providers (Hybrid Details) */}
              <div className="p-4 bg-slate-800/80 border border-slate-700 rounded-xl">
                <span className="text-[11px] text-slate-400 uppercase font-mono tracking-wider">
                  Model Providers (Hybrid)
                </span>
                <div className="mt-1 space-y-1">
                  <div className="flex items-center gap-1.5 text-xs font-mono">
                    <span className="text-slate-400">Enc:</span>
                    <span
                      className={`px-1.5 py-0.5 rounded font-bold uppercase ${
                        report.encoderProvider === "webgpu"
                          ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                          : "bg-blue-500/20 text-blue-300 border border-blue-500/40"
                      }`}
                    >
                      {report.encoderProvider}
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5 text-xs font-mono">
                    <span className="text-slate-400">Dec:</span>
                    <span className="px-1.5 py-0.5 rounded font-bold uppercase bg-blue-500/20 text-blue-300 border border-blue-500/40">
                      {report.decoderProvider}
                    </span>
                  </div>
                  {report.fallbackReason && (
                    <p className="text-[10px] text-amber-400 font-mono italic">
                      {report.fallbackReason}
                    </p>
                  )}
                </div>
              </div>

              {/* Number of Generated Tokens */}
              <div className="p-4 bg-slate-800/80 border border-slate-700 rounded-xl">
                <span className="text-[11px] text-slate-400 uppercase font-mono tracking-wider">
                  Generated Tokens
                </span>
                <div className="mt-1 flex items-baseline gap-1">
                  <span className="text-2xl font-bold font-mono text-white">
                    {report.numGeneratedTokens}
                  </span>
                  <span className="text-xs text-slate-400">tokens</span>
                </div>
                <div className="text-[10px] font-mono text-slate-400 mt-1">
                  Sequence Changed:{" "}
                  <span
                    className={
                      report.tokenSequenceChanged
                        ? "text-emerald-400 font-bold"
                        : "text-rose-400 font-bold"
                    }
                  >
                    {report.tokenSequenceChanged ? "YES" : "NO"}
                  </span>
                </div>
              </div>

              {/* Only Special Tokens? */}
              <div className="p-4 bg-slate-800/80 border border-slate-700 rounded-xl">
                <span className="text-[11px] text-slate-400 uppercase font-mono tracking-wider">
                  Only Special Tokens?
                </span>
                <div className="mt-1 flex items-center gap-2">
                  {report.isOnlySpecialTokens ? (
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-bold bg-rose-500/20 text-rose-400 border border-rose-500/40">
                      <AlertTriangle className="w-3 h-3 mr-1" />
                      YES (BLANK OUTPUT)
                    </span>
                  ) : (
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40">
                      <CheckCircle2 className="w-3 h-3 mr-1" />
                      NO (REAL CONTENT)
                    </span>
                  )}
                </div>
                <div className="text-[10px] font-mono text-slate-400 mt-1">
                  Start: {report.decoderStartTokenId} &bull; EOS: {report.eosTokenId}
                </div>
              </div>

              {/* Total Latency */}
              <div className="p-4 bg-slate-800/80 border border-slate-700 rounded-xl">
                <span className="text-[11px] text-slate-400 uppercase font-mono tracking-wider">
                  Inference Latency
                </span>
                <div className="mt-1 flex items-baseline gap-1">
                  <span className="text-2xl font-bold font-mono text-amber-300">
                    {report.timings.totalMs}
                  </span>
                  <span className="text-xs text-slate-400">ms</span>
                </div>
                <div className="text-[10px] font-mono text-slate-400 mt-1">
                  Enc: {report.timings.encoderMs}ms &bull; Dec: {report.timings.decoderMs}ms
                </div>
              </div>
            </div>

            {/* Decoded Text Panels */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* After Special Tokens (Clean Kannada) */}
              <div className="p-5 bg-slate-800/90 border-2 border-emerald-500/40 rounded-xl space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-bold text-emerald-400 uppercase tracking-wider flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4" />
                    Decoded Text (After Special Tokens Removed)
                  </h3>
                  <span className="text-[10px] text-slate-400 font-mono">
                    Length: {report.decodedTextAfterSpecialTokens.length} chars
                  </span>
                </div>
                <div className="p-4 bg-slate-950 rounded-lg border border-slate-800 min-h-[90px] flex items-center">
                  {report.decodedTextAfterSpecialTokens ? (
                    <p className="text-2xl font-medium text-emerald-200 tracking-wide font-sans">
                      {report.decodedTextAfterSpecialTokens}
                    </p>
                  ) : (
                    <p className="text-xs font-mono text-rose-400 italic">
                      [Empty String - Model generated no valid byte tokens]
                    </p>
                  )}
                </div>
              </div>

              {/* Before Special Tokens (Raw Tokens) */}
              <div className="p-5 bg-slate-800/90 border border-slate-700 rounded-xl space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                    <Info className="w-4 h-4 text-slate-400" />
                    Decoded Text (Before Removing Special Tokens)
                  </h3>
                  <span className="text-[10px] text-slate-400 font-mono">
                    Raw BPE token string
                  </span>
                </div>
                <div className="p-4 bg-slate-950 rounded-lg border border-slate-800 min-h-[90px] flex items-center">
                  <p className="text-sm font-mono text-amber-300/90 break-all">
                    {report.decodedTextBeforeSpecialTokens || "[No tokens]"}
                  </p>
                </div>
              </div>
            </div>

            {/* Generated Token IDs Badge Grid */}
            <div className="p-5 bg-slate-800/80 border border-slate-700 rounded-xl space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                  First {report.first30TokenIds.length} Generated Token IDs
                </h3>
                <span className="text-[11px] font-mono text-slate-400">
                  RoBERTa Vocab Range: 0 - 99,999
                </span>
              </div>
              <div className="flex flex-wrap gap-2 pt-1">
                {report.first30TokenIds.map((id, idx) => {
                  const isSpecial = id <= 4;
                  return (
                    <div
                      key={idx}
                      className={`px-2.5 py-1 rounded-lg text-xs font-mono border flex items-center gap-1.5 shadow-sm ${
                        isSpecial
                          ? "bg-rose-950/60 border-rose-700 text-rose-300"
                          : "bg-slate-900 border-amber-500/40 text-amber-300"
                      }`}
                    >
                      <span className="text-[10px] text-slate-500">#{idx}</span>
                      <span className="font-bold">{id}</span>
                      {isSpecial && (
                        <span className="text-[9px] uppercase tracking-wider text-rose-400 font-semibold">
                          ({id === 0 ? "<s>" : id === 1 ? "<pad>" : id === 2 ? "</s>" : id === 3 ? "<unk>" : "<mask>"})
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* ONNX Model Architecture Specs: Tensor Names & Shapes */}
            <div className="p-5 bg-slate-800/80 border border-slate-700 rounded-xl space-y-4">
              <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                Model Tensor Names & Shapes
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Vision Encoder Specs */}
                <div className="p-4 bg-slate-950 rounded-lg border border-slate-800 space-y-2">
                  <span className="text-xs font-bold text-amber-400 uppercase tracking-wide">
                    Vision Encoder ({report.encoderProvider.toUpperCase()})
                  </span>
                  <div className="space-y-1 text-xs font-mono">
                    <div className="text-slate-400">
                      Inputs:{" "}
                      <span className="text-slate-200">
                        {report.encoderInputNames.join(", ")}
                      </span>
                    </div>
                    {report.encoderInputShapes.map((shape, i) => (
                      <div key={i} className="text-emerald-400 pl-2">
                        &bull; {shape}
                      </div>
                    ))}
                    <div className="text-slate-400 pt-1">
                      Outputs:{" "}
                      <span className="text-slate-200">
                        {report.encoderOutputNames.join(", ")}
                      </span>
                    </div>
                    {report.encoderOutputShapes.map((shape, i) => (
                      <div key={i} className="text-emerald-400 pl-2">
                        &bull; {shape}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Autoregressive Decoder Specs */}
                <div className="p-4 bg-slate-950 rounded-lg border border-slate-800 space-y-2">
                  <span className="text-xs font-bold text-amber-400 uppercase tracking-wide">
                    Autoregressive Decoder ({report.decoderProvider.toUpperCase()})
                  </span>
                  <div className="space-y-1 text-xs font-mono">
                    <div className="text-slate-400">
                      Inputs:{" "}
                      <span className="text-slate-200">
                        {report.decoderInputNames.join(", ")}
                      </span>
                    </div>
                    {report.decoderInputShapes.map((shape, i) => (
                      <div key={i} className="text-emerald-400 pl-2">
                        &bull; {shape}
                      </div>
                    ))}
                    <div className="text-slate-400 pt-1">
                      Outputs:{" "}
                      <span className="text-slate-200">
                        {report.decoderOutputNames.join(", ")}
                      </span>
                    </div>
                    {report.decoderOutputShapes.map((shape, i) => (
                      <div key={i} className="text-emerald-400 pl-2">
                        &bull; {shape}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Detailed Step-by-Step Autoregressive Trace */}
            <div className="p-5 bg-slate-800/80 border border-slate-700 rounded-xl space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                  Step-by-Step Logit Trace (Input IDs &rarr; Output Dims &rarr; Predicted Token)
                </h3>
                <span className="text-[11px] font-mono text-slate-400">
                  Shows the exact int64 input sequence fed into the decoder at each step
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono border-collapse">
                  <thead>
                    <tr className="border-b border-slate-700 text-slate-400">
                      <th className="py-2 px-3">Step</th>
                      <th className="py-2 px-3">Decoder Input IDs</th>
                      <th className="py-2 px-3">Output Dims</th>
                      <th className="py-2 px-3">Chosen ID</th>
                      <th className="py-2 px-3">Raw BPE Token</th>
                      <th className="py-2 px-3">Max Logit</th>
                      <th className="py-2 px-3">Special?</th>
                      <th className="py-2 px-3">Top Candidates / Logits</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800">
                    {report.steps.map((step) => (
                      <tr key={step.step} className="hover:bg-slate-700/30">
                        <td className="py-2 px-3 text-slate-400">#{step.step}</td>
                        <td className="py-2 px-3 text-slate-300 max-w-[200px] truncate" title={`[${step.inputIds.join(", ")}]`}>
                          [{step.inputIds.join(", ")}]
                        </td>
                        <td className="py-2 px-3 text-cyan-400">
                          [{step.outputDims.join(", ")}]
                        </td>
                        <td className="py-2 px-3 font-bold text-amber-300">
                          {step.selectedTokenId}
                        </td>
                        <td className="py-2 px-3 text-emerald-300">
                          {step.selectedRawToken || "—"}
                        </td>
                        <td className="py-2 px-3 text-slate-300">{step.maxLogit}</td>
                        <td className="py-2 px-3">
                          {step.isSpecial ? (
                            <span className="text-[10px] font-bold text-rose-400 bg-rose-950/60 px-1.5 py-0.5 rounded border border-rose-800">
                              SPECIAL
                            </span>
                          ) : (
                            <span className="text-[10px] font-bold text-emerald-400 bg-emerald-950/60 px-1.5 py-0.5 rounded border border-emerald-800">
                              NORMAL
                            </span>
                          )}
                        </td>
                        <td className="py-2 px-3">
                          <div className="flex flex-wrap gap-1.5 max-w-[450px]">
                            {step.topCandidates.map((cand, ci) => (
                              <span
                                key={ci}
                                className={`text-[10px] px-1.5 py-0.5 rounded border ${
                                  cand.tokenId === step.selectedTokenId
                                    ? "bg-amber-500/20 text-amber-300 border-amber-500/40 font-bold"
                                    : "bg-slate-900 text-slate-400 border-slate-800"
                                }`}
                              >
                                {cand.tokenId} ({cand.rawToken || "sp"}): {cand.logit}
                              </span>
                            ))}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

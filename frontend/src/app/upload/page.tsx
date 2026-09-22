"use client";

import React, { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { uploadDocumentFile, saveBrowserOcrResult } from "@/lib/api";
import { DocumentUploadResponse } from "@/lib/types";
import { getBrowserTrOCR, BrowserOCRResult } from "@/lib/browserTrOCR";
import { DownloadProgress, computeSHA256 } from "@/lib/modelManager";
import {
  Upload,
  FileText,
  CheckCircle2,
  AlertCircle,
  ArrowRight,
  Info,
  Loader2,
  X,
  FileCheck,
  Cpu,
} from "lucide-react";

export default function UploadPage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const user = localStorage.getItem("auth_user");
      if (!user) {
        router.push("/login");
      }
    }
  }, [router]);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isHandwritten, setIsHandwritten] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessingLocal, setIsProcessingLocal] = useState(false);
  const [modelProgress, setModelProgress] = useState<DownloadProgress | null>(null);
  const [browserOcrResult, setBrowserOcrResult] = useState<BrowserOCRResult | null>(null);
  const [uploadResult, setUploadResult] = useState<DocumentUploadResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const allowedTypes = [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff"];

  const handleFileSelect = (file: File) => {
    const ext = "." + file.name.split(".").pop()?.toLowerCase();
    if (!allowedTypes.includes(ext)) {
      setErrorMessage(`Invalid file format '${ext}'. Please upload a PDF or supported image (PNG, JPG, WEBP, TIFF).`);
      return;
    }
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(file);
    if (/\.(png|jpe?g|webp|tiff|bmp)$/i.test(file.name)) {
      setPreviewUrl(URL.createObjectURL(file));
      setIsHandwritten(true);
    } else {
      setPreviewUrl(null);
    }
    setErrorMessage(null);
    setUploadResult(null);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleUploadSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;

    setErrorMessage(null);
    setUploadResult(null);

    try {
      const isImage = /\.(png|jpe?g|webp|tiff|bmp)$/i.test(selectedFile.name);

      // If Handwritten Kannada is selected and the file is an image, run browser TrOCR locally
      if (isHandwritten && isImage) {
        setIsProcessingLocal(true);
        setModelProgress({
          stage: "checking",
          loadedBytes: 0,
          totalBytes: 0,
          percentage: 0,
          message: "Checking OCR model in local cache...",
        });

        // 1. Download/initialize model
        const engine = await getBrowserTrOCR((p) => setModelProgress(p));

        // 2. Processing image progress
        setModelProgress({
          stage: "checking",
          loadedBytes: 100,
          totalBytes: 100,
          percentage: 100,
          message: "Processing image...",
        });

        // 3. Run OCR locally with safety timeout ceiling (5 minutes so legitimate 1-2 min operations on slower CPUs are never cancelled)
        let timeoutId: any;
        const timeoutPromise = new Promise<never>((_, reject) => {
          timeoutId = setTimeout(
            () => reject(new Error("OCR processing timed out. If your device is running on CPU/WASM, please try a smaller crop or enable WebGPU.")),
            300000
          );
        });

        console.log(`[Audit:Stage1-Upload] Starting local handwritten OCR for: ${selectedFile.name} (${selectedFile.size} bytes)`);

        const localResult = await Promise.race([
          engine.recognize(selectedFile).finally(() => clearTimeout(timeoutId)),
          timeoutPromise,
        ]);

        console.log(`[Audit:Stage1-Upload] OCR result received from engine:`, {
          has_text: Boolean(localResult.text && localResult.text.trim()),
          text_length: localResult.text ? localResult.text.length : 0,
          confidence: localResult.confidence,
          provider: localResult.executionProvider,
          latency_ms: localResult.latencyMs,
          token_count: localResult.tokens?.length || 0,
        });

        setModelProgress({
          stage: "ready",
          loadedBytes: 100,
          totalBytes: 100,
          percentage: 100,
          message: "OCR completed",
        });

        // 4. Calculate file SHA-256 hash and encode file as base64 for storage persistence
        const fileBuffer = await selectedFile.arrayBuffer();
        const fileHash = await computeSHA256(fileBuffer);

        // Convert ArrayBuffer to base64 for backend image storage
        const uint8Arr = new Uint8Array(fileBuffer);
        let binaryStr = "";
        const chunkSize = 8192;
        for (let i = 0; i < uint8Arr.length; i += chunkSize) {
          binaryStr += String.fromCharCode(...uint8Arr.subarray(i, i + chunkSize));
        }
        const fileBase64 = btoa(binaryStr);

        const payload = {
          filename: selectedFile.name,
          file_hash: fileHash,
          file_size: selectedFile.size,
          text: localResult.text,
          confidence: localResult.confidence,
          execution_provider: localResult.executionProvider,
          latency_ms: localResult.latencyMs,
          tokens: localResult.tokens,
          file_base64: fileBase64,
        };

        console.log(`[Audit:Stage4-RequestPayload] Submitting POST /api/v1/documents/browser-result:`, {
          field_names: Object.keys(payload),
          has_text: Boolean(payload.text && payload.text.trim()),
          text_length: payload.text.length,
          confidence: payload.confidence,
          provider: payload.execution_provider,
          hash_prefix: payload.file_hash.substring(0, 8),
        });

        // 5. Save original document metadata and browser OCR output through lightweight backend endpoint
        let savedResult: DocumentUploadResponse;
        try {
          savedResult = await saveBrowserOcrResult(payload);
          console.log(`[Audit:Stage4-RequestPayload] Backend save response received:`, {
            document_id: savedResult.document.id,
            status: savedResult.document.status,
            is_duplicate: savedResult.is_duplicate,
            task_id: savedResult.task_id,
          });
        } catch (saveErr: any) {
          throw new Error(`Saving the local OCR result failed: ${saveErr.message || saveErr}`);
        }

        // 6. Show final result and navigate to document page
        setBrowserOcrResult(localResult);
        setUploadResult(savedResult);
        setIsProcessingLocal(false);

        if (savedResult?.document?.id) {
          router.push(`/documents/${savedResult.document.id}`);
        }
        return;
      }

      setIsUploading(true);
      const result = await uploadDocumentFile(selectedFile, isHandwritten);
      setUploadResult(result);
      if (result?.document?.id) {
        router.push(`/documents/${result.document.id}`);
      }
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to process and upload document.");
    } finally {
      setIsProcessingLocal(false);
      setIsUploading(false);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      {/* Page Title */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight">Upload Document</h1>
        <p className="text-sm text-slate-500 mt-1">
          Upload PDF invoices, receipts, or images for asynchronous SHA-256 deduplication, MinIO storage, and AI extraction.
        </p>
      </div>

      {/* Error Banner */}
      {errorMessage && (
        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
          <div className="flex-1 text-sm font-medium">
            <p className="font-semibold">Upload Error</p>
            <p className="mt-0.5">{errorMessage}</p>
          </div>
          <button onClick={() => setErrorMessage(null)} className="text-rose-500 hover:text-rose-700">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Duplicate / Success Result Card */}
      {uploadResult && (
        <div
          className={`p-5 rounded-xl border shadow-sm ${
            uploadResult.is_duplicate
              ? "bg-amber-50 border-amber-200 text-amber-900"
              : "bg-emerald-50 border-emerald-200 text-emerald-900"
          }`}
        >
          <div className="flex items-start gap-3.5">
            {uploadResult.is_duplicate ? (
              <Info className="w-6 h-6 text-amber-600 shrink-0 mt-0.5" />
            ) : (
              <CheckCircle2 className="w-6 h-6 text-emerald-600 shrink-0 mt-0.5" />
            )}
            <div className="flex-1 space-y-2">
              <div>
                <h3 className="text-base font-bold">
                  {uploadResult.is_duplicate ? "Duplicate Document Detected" : "Document Ingestion Successful"}
                </h3>
                <p className="text-sm opacity-90">{uploadResult.message}</p>
              </div>

              <div className="bg-white/80 p-3 rounded-lg border border-black/5 text-xs space-y-1 font-mono">
                <p>
                  <span className="font-sans font-semibold text-slate-500">Document ID:</span> #{uploadResult.document.id}
                </p>
                <p>
                  <span className="font-sans font-semibold text-slate-500">Filename:</span> {uploadResult.document.filename}
                </p>
                <p>
                  <span className="font-sans font-semibold text-slate-500">SHA-256:</span> {uploadResult.document.file_hash.substring(0, 24)}...
                </p>
                {uploadResult.task_id && (
                  <p>
                    <span className="font-sans font-semibold text-slate-500">
                      {uploadResult.task_id.startsWith("local") ? "Processing Job:" : "Celery Task:"}
                    </span>{" "}
                    {uploadResult.task_id}
                  </p>
                )}
              </div>

              <div className="pt-2 flex items-center gap-3">
                <Link
                  href={`/documents/${uploadResult.document.id}`}
                  className="inline-flex items-center gap-1.5 px-4 py-2 text-sm font-semibold text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 shadow-sm transition-colors"
                >
                  <span>View Document & Extracted Results</span>
                  <ArrowRight className="w-4 h-4" />
                </Link>
                <button
                  onClick={() => {
                    setUploadResult(null);
                    setSelectedFile(null);
                  }}
                  className="text-xs font-semibold text-slate-600 hover:text-slate-900 underline"
                >
                  Upload Another File
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Upload Box Form */}
      <form onSubmit={handleUploadSubmit} className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-5">
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.png,.jpg,.jpeg,.webp,.tiff"
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              handleFileSelect(e.target.files[0]);
            }
          }}
        />

        {/* Dropzone */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
            isDragging
              ? "border-indigo-500 bg-indigo-50/50 scale-[0.99]"
              : selectedFile
              ? "border-emerald-400 bg-emerald-50/30"
              : "border-slate-300 hover:border-indigo-400 hover:bg-slate-50/50"
          }`}
        >
          <div className="w-12 h-12 mx-auto rounded-full bg-indigo-50 text-indigo-600 flex items-center justify-center mb-3">
            <Upload className="w-6 h-6" />
          </div>

          <p className="text-sm font-semibold text-slate-800">
            {selectedFile ? "Click or drop to replace selected file" : "Click to select a file, or drag and drop"}
          </p>
          <p className="text-xs text-slate-500 mt-1">Supported formats: PDF, PNG, JPG, JPEG, WEBP, TIFF (up to 50MB)</p>
        </div>

        {/* Selected File Details Box */}
        {selectedFile && (
          <div className="flex items-center justify-between p-3.5 bg-slate-50 border border-slate-200 rounded-lg">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-indigo-100 text-indigo-700 flex items-center justify-center">
                <FileCheck className="w-5 h-5" />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-900 truncate max-w-xs">{selectedFile.name}</p>
                <p className="text-xs text-slate-500">{formatFileSize(selectedFile.size)}</p>
              </div>
            </div>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setSelectedFile(null);
              }}
              className="p-1 text-slate-400 hover:text-slate-600 rounded-md"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Handwritten Kannada In-Browser OCR Controls */}
        <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <input
                id="handwritten-toggle"
                type="checkbox"
                checked={isHandwritten}
                onChange={(e) => {
                  setIsHandwritten(e.target.checked);
                  if (!e.target.checked) {
                    setBrowserOcrResult(null);
                    setModelProgress(null);
                  }
                }}
                className="w-4 h-4 text-indigo-600 rounded border-slate-300 focus:ring-indigo-500 cursor-pointer"
              />
              <label htmlFor="handwritten-toggle" className="text-sm font-semibold text-slate-800 cursor-pointer">
                Handwritten Kannada Document (In-Browser TrOCR)
              </label>
            </div>
            <span
              className={`px-2.5 py-0.5 text-xs font-semibold rounded-full border ${
                isHandwritten
                  ? "bg-indigo-50 text-indigo-700 border-indigo-200"
                  : "bg-slate-100 text-slate-600 border-slate-200"
              }`}
            >
              {isHandwritten ? "Client Execution" : "Server Routing"}
            </span>
          </div>
          <p className="text-xs text-slate-500">
            When enabled, the IIT Bombay Indic-TrOCR model automatically runs locally in your browser using WebGPU/WASM without consuming server memory.
          </p>

          {/* Model Download & Verification Progress Bar */}
          {modelProgress && (isProcessingLocal || modelProgress.stage !== "ready") && (
            <div className="pt-2 space-y-1.5 border-t border-slate-200">
              <div className="flex items-center justify-between text-xs">
                <span className="font-medium text-slate-700">{modelProgress.message}</span>
                <span className="font-mono text-slate-500">{modelProgress.percentage}%</span>
              </div>
              <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-indigo-600 transition-all duration-200"
                  style={{ width: `${modelProgress.percentage}%` }}
                />
              </div>
            </div>
          )}

          {/* Browser OCR Result Banner */}
          {browserOcrResult && (
            <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-lg text-xs space-y-1">
              <div className="flex items-center justify-between">
                <span className="font-bold text-emerald-900">Recognized Kannada Text (Browser TrOCR)</span>
                <span className="font-mono text-emerald-700">
                  {browserOcrResult.executionProvider.toUpperCase()} • {browserOcrResult.latencyMs}ms • {(browserOcrResult.confidence * 100).toFixed(1)}% conf
                </span>
              </div>
              <p className="font-medium text-slate-900 text-sm bg-white/80 p-2 rounded border border-emerald-100 font-sans">
                {browserOcrResult.text || "—"}
              </p>
            </div>
          )}
        </div>

        {/* Automatic Hybrid Script & Style Routing Indicator */}
        <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg flex items-center justify-between">
          <div>
            <span className="text-sm font-semibold text-slate-800">Automatic Script & Style Routing</span>
            <p className="text-xs text-slate-500">
              Per-region layout routing automatically directs printed Kannada to EasyOCR and handwritten Kannada to IIT Bombay Indic-TrOCR v0.0.2.
            </p>
          </div>
          <span className="px-2.5 py-1 text-xs font-semibold rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 shrink-0 ml-3">
            Automatic
          </span>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={!selectedFile || isUploading || isProcessingLocal}
          className="w-full py-3 px-4 rounded-xl text-sm font-semibold text-white bg-indigo-600 hover:bg-indigo-700 shadow-md shadow-indigo-100 transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isProcessingLocal ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>{modelProgress?.message || "Running Browser OCR..."}</span>
            </>
          ) : isUploading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Uploading & Processing...</span>
            </>
          ) : (
            <>
              <Upload className="w-4 h-4" />
              <span>Upload Document</span>
            </>
          )}
        </button>
      </form>
    </div>
  );
}

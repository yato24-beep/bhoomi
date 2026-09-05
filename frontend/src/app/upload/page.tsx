"use client";

import React, { useState, useRef } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { uploadDocumentFile } from "@/lib/api";
import { DocumentUploadResponse } from "@/lib/types";
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
} from "lucide-react";

export default function UploadPage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<DocumentUploadResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const allowedTypes = [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff"];

  const handleFileSelect = (file: File) => {
    const ext = "." + file.name.split(".").pop()?.toLowerCase();
    if (!allowedTypes.includes(ext)) {
      setErrorMessage(`Invalid file format '${ext}'. Please upload a PDF or supported image (PNG, JPG, WEBP, TIFF).`);
      return;
    }
    setSelectedFile(file);
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

    setIsUploading(true);
    setErrorMessage(null);
    setUploadResult(null);

    try {
      const result = await uploadDocumentFile(selectedFile);
      setUploadResult(result);
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to upload file to backend.");
    } finally {
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
                    <span className="font-sans font-semibold text-slate-500">Celery Task:</span> {uploadResult.task_id}
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

        {/* Submit Button */}
        <button
          type="submit"
          disabled={!selectedFile || isUploading}
          className="w-full py-3 px-4 rounded-xl text-sm font-semibold text-white bg-indigo-600 hover:bg-indigo-700 shadow-md shadow-indigo-100 transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isUploading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Uploading to MinIO & Queuing Task...</span>
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

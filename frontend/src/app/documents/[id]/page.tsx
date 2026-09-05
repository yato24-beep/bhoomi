"use client";

import React, { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import {
  fetchDocument,
  fetchDocumentStatus,
  fetchDocumentResults,
  fetchDocumentFields,
  deleteDocumentRecord,
  getDirectDownloadUrl,
} from "@/lib/api";
import {
  DocumentItem,
  ExtractionResult,
  ExtractedFieldsSummary,
} from "@/lib/types";
import { StatusBadge } from "@/components/status-badge";
import {
  ArrowLeft,
  Download,
  Trash2,
  RefreshCw,
  FileText,
  Copy,
  Check,
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  Loader2,
  Table,
  Code2,
  Layers,
} from "lucide-react";

export default function DocumentDetailsPage() {
  const params = useParams();
  const router = useRouter();
  const documentId = Number(params.id);

  const [document, setDocument] = useState<DocumentItem | null>(null);
  const [results, setResults] = useState<ExtractionResult | null>(null);
  const [fieldsSummary, setFieldsSummary] = useState<ExtractedFieldsSummary | null>(null);
  const [activeTab, setActiveTab] = useState<"fields" | "json">("fields");
  const [loading, setLoading] = useState(true);
  const [copiedJson, setCopiedJson] = useState(false);
  const [minConfidenceFilter, setMinConfidenceFilter] = useState<number | undefined>(undefined);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    if (!documentId) return;
    try {
      setErrorMessage(null);
      const doc = await fetchDocument(documentId);
      setDocument(doc);

      if (doc.status === "COMPLETED") {
        const [res, fields] = await Promise.all([
          fetchDocumentResults(documentId),
          fetchDocumentFields(documentId, minConfidenceFilter),
        ]);
        setResults(res);
        setFieldsSummary(fields);
      }
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to load document details");
    } finally {
      setLoading(false);
    }
  }, [documentId, minConfidenceFilter]);

  useEffect(() => {
    loadData();

    // Auto-poll if document is in processing state
    const interval = setInterval(async () => {
      if (document && (document.status === "UPLOADED" || document.status === "PROCESSING")) {
        const statusRes = await fetchDocumentStatus(documentId);
        if (statusRes.status !== document.status) {
          loadData();
        }
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [documentId, loadData, document]);

  const handleDelete = async () => {
    if (!document || !confirm(`Delete document "${document.filename}"?`)) return;
    try {
      await deleteDocumentRecord(document.id);
      router.push("/");
    } catch (err: any) {
      alert(`Error deleting document: ${err.message}`);
    }
  };

  const handleCopyJson = () => {
    if (!results) return;
    navigator.clipboard.writeText(JSON.stringify(results.extracted_data, null, 2));
    setCopiedJson(true);
    setTimeout(() => setCopiedJson(false), 2000);
  };

  if (loading && !document) {
    return (
      <div className="py-24 text-center">
        <Loader2 className="w-8 h-8 animate-spin mx-auto text-indigo-600 mb-3" />
        <p className="text-sm font-medium text-slate-500">Loading document #{documentId}...</p>
      </div>
    );
  }

  if (errorMessage && !document) {
    return (
      <div className="max-w-xl mx-auto py-16 text-center space-y-4">
        <AlertTriangle className="w-12 h-12 text-rose-500 mx-auto" />
        <h2 className="text-lg font-bold text-slate-900">Document Not Found</h2>
        <p className="text-sm text-slate-500">{errorMessage}</p>
        <Link
          href="/"
          className="inline-flex items-center gap-2 px-4 py-2 text-sm font-semibold text-white bg-indigo-600 rounded-lg hover:bg-indigo-700"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Dashboard
        </Link>
      </div>
    );
  }

  if (!document) return null;

  const isProcessing = document.status === "UPLOADED" || document.status === "PROCESSING";

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Top Breadcrumb & Action Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Link
            href="/"
            className="p-2 text-slate-500 hover:text-slate-900 hover:bg-slate-200/60 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </Link>
          <div>
            <div className="flex items-center gap-2.5">
              <h1 className="text-2xl font-bold text-slate-900 tracking-tight">{document.filename}</h1>
              <StatusBadge status={document.status} />
            </div>
            <p className="text-xs font-mono text-slate-400 mt-0.5">
              ID #{document.id} &bull; SHA-256: {document.file_hash}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={loadData}
            className="p-2 text-slate-600 hover:bg-slate-100 border border-slate-300 rounded-lg transition-colors"
            title="Refresh Data"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          <a
            href={getDirectDownloadUrl(document.id)}
            className="flex items-center gap-2 px-3.5 py-2 text-sm font-medium text-slate-700 bg-white border border-slate-300 rounded-lg hover:bg-slate-50 shadow-sm transition-colors"
          >
            <Download className="w-4 h-4 text-slate-500" />
            Download File
          </a>
          <button
            onClick={handleDelete}
            className="flex items-center gap-2 px-3.5 py-2 text-sm font-medium text-rose-600 bg-white border border-rose-200 rounded-lg hover:bg-rose-50 transition-colors"
          >
            <Trash2 className="w-4 h-4" />
            Delete
          </button>
        </div>
      </div>

      {/* Live Processing Animation Banner */}
      {isProcessing && (
        <div className="p-6 bg-gradient-to-r from-indigo-50 to-blue-50 border border-indigo-100 rounded-2xl shadow-sm text-center space-y-3">
          <Loader2 className="w-8 h-8 text-indigo-600 animate-spin mx-auto" />
          <h3 className="text-base font-bold text-slate-900">
            {document.status === "UPLOADED"
              ? "Document Queued in Celery"
              : "Celery Worker Processing Document..."}
          </h3>
          <p className="text-xs text-slate-600 max-w-md mx-auto">
            Retrieving file from MinIO, executing text extraction, computing bounding boxes, and validating business rules. Results will appear automatically.
          </p>
          <div className="w-48 h-1.5 bg-indigo-200 rounded-full overflow-hidden mx-auto">
            <div className="w-full h-full bg-indigo-600 animate-[pulse_1.5s_infinite]" />
          </div>
        </div>
      )}

      {/* Metadata Card */}
      <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
        <div>
          <span className="font-semibold text-slate-500">MinIO Storage Key</span>
          <p className="font-mono text-slate-800 mt-0.5 truncate">{document.storage_path || "N/A"}</p>
        </div>
        <div>
          <span className="font-semibold text-slate-500">Ingestion Timestamp</span>
          <p className="text-slate-800 mt-0.5">{new Date(document.created_at).toLocaleString()}</p>
        </div>
        <div>
          <span className="font-semibold text-slate-500">Overall Accuracy</span>
          <p className="text-slate-800 mt-0.5 font-semibold">
            {results ? `${(results.confidence_score * 100).toFixed(0)}% Confidence` : "Pending completion"}
          </p>
        </div>
      </div>

      {/* Completed Results Section */}
      {document.status === "COMPLETED" && (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden space-y-4">
          {/* Tab Bar */}
          <div className="flex items-center justify-between border-b border-slate-200 px-6 pt-4">
            <div className="flex gap-6">
              <button
                onClick={() => setActiveTab("fields")}
                className={`flex items-center gap-2 pb-3.5 text-sm font-semibold border-b-2 transition-all ${
                  activeTab === "fields"
                    ? "border-indigo-600 text-indigo-600"
                    : "border-transparent text-slate-500 hover:text-slate-800"
                }`}
              >
                <Table className="w-4 h-4" />
                Extracted Fields ({fieldsSummary?.total_fields || 0})
              </button>
              <button
                onClick={() => setActiveTab("json")}
                className={`flex items-center gap-2 pb-3.5 text-sm font-semibold border-b-2 transition-all ${
                  activeTab === "json"
                    ? "border-indigo-600 text-indigo-600"
                    : "border-transparent text-slate-500 hover:text-slate-800"
                }`}
              >
                <Code2 className="w-4 h-4" />
                Structured JSON & Validation
              </button>
            </div>

            {activeTab === "fields" && (
              <div className="flex items-center gap-2 pb-2">
                <span className="text-xs text-slate-500 font-medium hidden sm:inline">Filter:</span>
                <button
                  onClick={() => setMinConfidenceFilter(undefined)}
                  className={`px-2.5 py-1 rounded-md text-xs font-semibold ${
                    minConfidenceFilter === undefined
                      ? "bg-indigo-600 text-white"
                      : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                  }`}
                >
                  All
                </button>
                <button
                  onClick={() => setMinConfidenceFilter(0.95)}
                  className={`px-2.5 py-1 rounded-md text-xs font-semibold ${
                    minConfidenceFilter === 0.95
                      ? "bg-emerald-600 text-white"
                      : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                  }`}
                >
                  ≥ 95% Conf
                </button>
              </div>
            )}
          </div>

          {/* TAB 1: Extracted Fields Table */}
          {activeTab === "fields" && (
            <div className="p-6 pt-2 space-y-4">
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm text-slate-600">
                  <thead className="bg-slate-50 text-slate-700 font-semibold border-b border-slate-200 text-xs uppercase tracking-wider">
                    <tr>
                      <th className="py-3 px-4">Field Name</th>
                      <th className="py-3 px-4">Original OCR Value</th>
                      <th className="py-3 px-4">Normalized Value</th>
                      <th className="py-3 px-4">Confidence</th>
                      <th className="py-3 px-4">Page</th>
                      <th className="py-3 px-4">Bounding Box (x, y, w, h)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {fieldsSummary?.fields.map((field) => (
                      <tr key={field.id} className="hover:bg-slate-50/80 transition-colors">
                        <td className="py-3 px-4 font-semibold text-slate-900">{field.field_name}</td>
                        <td className="py-3 px-4 text-slate-700 font-mono text-xs">
                          {field.original_value || "—"}
                        </td>
                        <td className="py-3 px-4">
                          <span className="inline-block px-2 py-0.5 rounded bg-slate-100 font-medium text-xs text-indigo-700">
                            {field.normalized_value || "—"}
                          </span>
                        </td>
                        <td className="py-3 px-4">
                          <div className="flex items-center gap-2">
                            <div className="w-12 bg-slate-100 h-2 rounded-full overflow-hidden">
                              <div
                                className={`h-full rounded-full ${
                                  field.confidence_score >= 0.95
                                    ? "bg-emerald-500"
                                    : field.confidence_score >= 0.8
                                    ? "bg-amber-500"
                                    : "bg-rose-500"
                                }`}
                                style={{ width: `${field.confidence_score * 100}%` }}
                              />
                            </div>
                            <span className="text-xs font-semibold text-slate-700">
                              {(field.confidence_score * 100).toFixed(0)}%
                            </span>
                          </div>
                        </td>
                        <td className="py-3 px-4 text-xs font-semibold text-slate-500">
                          p.{field.source_page}
                        </td>
                        <td className="py-3 px-4 font-mono text-[11px] text-slate-400">
                          {field.bounding_box ? (
                            <span>
                              [{field.bounding_box.x_min.toFixed(2)}, {field.bounding_box.y_min.toFixed(2)},{" "}
                              {field.bounding_box.x_max.toFixed(2)}, {field.bounding_box.y_max.toFixed(2)}]
                            </span>
                          ) : (
                            "—"
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* TAB 2: Structured JSON & Validation */}
          {activeTab === "json" && results && (
            <div className="p-6 pt-2 space-y-6">
              {/* Validation Checklist */}
              <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl space-y-3">
                <div className="flex items-center gap-2 text-sm font-bold text-slate-900">
                  <ShieldCheck className="w-5 h-5 text-emerald-600" />
                  Business Validation Rules (
                  {results.validation_info.checks_passed?.length || 0} passed)
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                  {results.validation_info.checks_passed?.map((check, idx) => (
                    <div key={idx} className="flex items-center gap-2 text-slate-700">
                      <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                      <span>{check}</span>
                    </div>
                  ))}
                  {results.validation_info.errors?.map((err, idx) => (
                    <div key={idx} className="flex items-center gap-2 text-rose-700 font-semibold">
                      <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0" />
                      <span>{err}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* JSON Code Viewer */}
              <div className="relative">
                <div className="flex items-center justify-between pb-2">
                  <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                    Structured Output Payload
                  </span>
                  <button
                    onClick={handleCopyJson}
                    className="flex items-center gap-1.5 px-3 py-1 bg-slate-100 hover:bg-slate-200 rounded text-xs font-medium text-slate-700 transition-colors"
                  >
                    {copiedJson ? (
                      <>
                        <Check className="w-3.5 h-3.5 text-emerald-600" />
                        <span>Copied!</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5" />
                        <span>Copy JSON</span>
                      </>
                    )}
                  </button>
                </div>
                <pre className="p-4 bg-slate-900 text-slate-100 rounded-xl text-xs font-mono overflow-x-auto max-h-[500px]">
                  {JSON.stringify(results.extracted_data, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

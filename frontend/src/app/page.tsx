"use client";

import React, { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { DocumentItem, DocumentSearchItem } from "@/lib/types";
import { fetchDocuments, searchDocuments, deleteDocumentRecord, getDirectDownloadUrl } from "@/lib/api";
import { StatusBadge } from "@/components/status-badge";
import {
  FileText,
  Upload,
  RefreshCw,
  Search,
  ExternalLink,
  Download,
  Trash2,
  CheckCircle2,
  Clock,
  Loader2,
  AlertTriangle,
  Tag,
  X,
} from "lucide-react";

export default function DashboardPage() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [searchResults, setSearchResults] = useState<DocumentSearchItem[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("ALL");
  const [deletingId, setDeletingId] = useState<number | null>(null);

  const loadDocuments = useCallback(async () => {
    try {
      setError(null);
      const data = await fetchDocuments();
      setDocuments(data);
    } catch (err: any) {
      setError(err.message || "Failed to load documents from backend.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDocuments();

    // Auto-poll every 3 seconds if any document is in non-terminal state
    const interval = setInterval(() => {
      const hasActiveJobs = documents.some(
        (doc) => doc.status === "UPLOADED" || doc.status === "PROCESSING"
      );
      if (hasActiveJobs && !searchTerm.trim()) {
        loadDocuments();
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [loadDocuments, documents, searchTerm]);

  // Debounced backend search for filename and extracted fields
  useEffect(() => {
    if (!searchTerm.trim()) {
      setSearchResults(null);
      setSearching(false);
      return;
    }

    const timer = setTimeout(async () => {
      setSearching(true);
      try {
        const filter = statusFilter !== "ALL" ? statusFilter : undefined;
        const res = await searchDocuments(searchTerm.trim(), undefined, filter);
        setSearchResults(res.results);
      } catch (err: any) {
        console.error("Search error:", err);
      } finally {
        setSearching(false);
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [searchTerm, statusFilter]);

  const handleDelete = async (id: number, filename: string) => {
    if (!confirm(`Are you sure you want to delete "${filename}"?`)) return;
    try {
      setDeletingId(id);
      await deleteDocumentRecord(id);
      setDocuments((prev) => prev.filter((d) => d.id !== id));
      if (searchResults) {
        setSearchResults((prev) => prev?.filter((d) => d.id !== id) || null);
      }
    } catch (err: any) {
      alert(`Error deleting document: ${err.message}`);
    } finally {
      setDeletingId(null);
    }
  };

  const totalCount = documents.length;
  const completedCount = documents.filter((d) => d.status === "COMPLETED").length;
  const processingCount = documents.filter(
    (d) => d.status === "PROCESSING" || d.status === "UPLOADED"
  ).length;
  const failedCount = documents.filter((d) => d.status === "FAILED").length;

  const displayList = searchResults
    ? searchResults
    : statusFilter === "ALL"
    ? documents
    : documents.filter((d) => d.status === statusFilter);

  return (
    <div className="space-y-8">
      {/* Header Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">Documents Dashboard</h1>
          <p className="text-sm text-slate-500 mt-1">
            Search documents across filenames and extracted fields (vendors, amounts, invoice numbers).
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={loadDocuments}
            disabled={loading}
            className="flex items-center gap-2 px-3.5 py-2 text-sm font-medium text-slate-700 bg-white border border-slate-300 rounded-lg hover:bg-slate-50 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </button>
          <Link
            href="/upload"
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 shadow-sm transition-colors"
          >
            <Upload className="w-4 h-4" />
            Upload Document
          </Link>
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex items-center justify-between">
          <div>
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Total Documents</p>
            <p className="text-2xl font-bold text-slate-900 mt-1">{totalCount}</p>
          </div>
          <div className="w-10 h-10 rounded-lg bg-slate-100 flex items-center justify-center text-slate-600">
            <FileText className="w-5 h-5" />
          </div>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex items-center justify-between">
          <div>
            <p className="text-xs font-semibold text-emerald-600 uppercase tracking-wider">Completed</p>
            <p className="text-2xl font-bold text-slate-900 mt-1">{completedCount}</p>
          </div>
          <div className="w-10 h-10 rounded-lg bg-emerald-50 flex items-center justify-center text-emerald-600">
            <CheckCircle2 className="w-5 h-5" />
          </div>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex items-center justify-between">
          <div>
            <p className="text-xs font-semibold text-amber-600 uppercase tracking-wider">Processing</p>
            <p className="text-2xl font-bold text-slate-900 mt-1">{processingCount}</p>
          </div>
          <div className="w-10 h-10 rounded-lg bg-amber-50 flex items-center justify-center text-amber-600">
            <Loader2 className={`w-5 h-5 ${processingCount > 0 ? "animate-spin" : ""}`} />
          </div>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex items-center justify-between">
          <div>
            <p className="text-xs font-semibold text-rose-600 uppercase tracking-wider">Failed</p>
            <p className="text-2xl font-bold text-slate-900 mt-1">{failedCount}</p>
          </div>
          <div className="w-10 h-10 rounded-lg bg-rose-50 flex items-center justify-center text-rose-600">
            <AlertTriangle className="w-5 h-5" />
          </div>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 text-rose-600" />
            <p className="text-sm font-medium">{error}</p>
          </div>
          <button onClick={loadDocuments} className="text-xs font-semibold text-rose-800 underline hover:no-underline">
            Retry
          </button>
        </div>
      )}

      {/* Search & Filter Controls */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="w-5 h-5 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search by filename or extracted fields (e.g. Acme, 1350, INV-2026)..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-10 py-2.5 bg-white border border-slate-300 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 shadow-sm"
          />
          {searchTerm && (
            <button
              onClick={() => setSearchTerm("")}
              className="absolute right-3.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Status Filter Buttons */}
        <div className="flex items-center gap-1.5 bg-white p-1 rounded-xl border border-slate-300 shadow-sm text-xs font-semibold">
          {["ALL", "COMPLETED", "PROCESSING", "FAILED"].map((st) => (
            <button
              key={st}
              onClick={() => setStatusFilter(st)}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                statusFilter === st
                  ? "bg-indigo-600 text-white shadow-xs"
                  : "text-slate-600 hover:bg-slate-100"
              }`}
            >
              {st}
            </button>
          ))}
        </div>
      </div>

      {/* Search Status Indicator */}
      {searchTerm && (
        <div className="text-xs text-slate-500 flex items-center justify-between">
          <p>
            {searching ? (
              <span className="flex items-center gap-1.5 text-indigo-600 font-medium">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                Searching PostgreSQL database...
              </span>
            ) : (
              <span>
                Found <strong className="text-slate-900">{displayList.length}</strong> results for &ldquo;{searchTerm}&rdquo;
              </span>
            )}
          </p>
        </div>
      )}

      {/* Documents Table */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        {loading && documents.length === 0 ? (
          <div className="py-16 text-center text-slate-500">
            <Loader2 className="w-8 h-8 animate-spin mx-auto text-indigo-600 mb-3" />
            <p className="text-sm font-medium">Connecting to FastAPI and loading documents...</p>
          </div>
        ) : displayList.length === 0 ? (
          <div className="py-16 text-center text-slate-500">
            <FileText className="w-12 h-12 mx-auto text-slate-300 mb-3" />
            <p className="text-base font-semibold text-slate-800">No documents found</p>
            <p className="text-sm text-slate-500 mt-1 max-w-sm mx-auto">
              {searchTerm
                ? `No documents matched "${searchTerm}". Try searching for a different vendor, invoice number, or filename.`
                : "Upload your first PDF or image to start asynchronous extraction."}
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm text-slate-600">
              <thead className="bg-slate-50 text-slate-700 font-semibold border-b border-slate-200 text-xs uppercase tracking-wider">
                <tr>
                  <th className="py-3.5 px-4">ID</th>
                  <th className="py-3.5 px-4">Document & Matches</th>
                  <th className="py-3.5 px-4">SHA-256 Hash</th>
                  <th className="py-3.5 px-4">Status</th>
                  <th className="py-3.5 px-4">Uploaded At</th>
                  <th className="py-3.5 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-normal">
                {displayList.map((doc) => {
                  const searchItem = "matched_fields" in doc ? (doc as DocumentSearchItem) : null;
                  return (
                    <tr key={doc.id} className="hover:bg-slate-50/80 transition-colors">
                      <td className="py-3.5 px-4 font-semibold text-slate-900">#{doc.id}</td>
                      <td className="py-3.5 px-4">
                        <div className="space-y-1">
                          <Link
                            href={`/documents/${doc.id}`}
                            className="font-medium text-indigo-600 hover:text-indigo-800 flex items-center gap-1.5"
                          >
                            <FileText className="w-4 h-4 text-slate-400" />
                            {doc.filename}
                          </Link>

                          {/* Matched Fields Provenance Badges */}
                          {searchItem && searchItem.matched_fields && searchItem.matched_fields.length > 0 && (
                            <div className="flex flex-wrap gap-1.5 pt-0.5">
                              {searchItem.matched_fields.slice(0, 3).map((f) => (
                                <span
                                  key={f.id}
                                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-medium bg-amber-50 text-amber-800 border border-amber-200"
                                >
                                  <Tag className="w-3 h-3 text-amber-600" />
                                  <span className="font-semibold">{f.field_name}:</span> {f.normalized_value || f.original_value}
                                </span>
                              ))}
                              {searchItem.matched_fields.length > 3 && (
                                <span className="text-[11px] text-slate-400 font-medium self-center">
                                  +{searchItem.matched_fields.length - 3} more
                                </span>
                              )}
                            </div>
                          )}
                        </div>
                      </td>
                      <td className="py-3.5 px-4 font-mono text-xs text-slate-500">
                        {doc.file_hash.substring(0, 16)}...
                      </td>
                      <td className="py-3.5 px-4">
                        <StatusBadge status={doc.status} />
                      </td>
                      <td className="py-3.5 px-4 text-xs text-slate-500">
                        {new Date(doc.created_at).toLocaleString()}
                      </td>
                      <td className="py-3.5 px-4 text-right">
                        <div className="flex items-center justify-end gap-2">
                          <Link
                            href={`/documents/${doc.id}`}
                            className="p-1.5 text-slate-600 hover:text-indigo-600 hover:bg-indigo-50 rounded-md transition-colors"
                            title="View Details & Extracted Results"
                          >
                            <ExternalLink className="w-4 h-4" />
                          </Link>
                          <a
                            href={getDirectDownloadUrl(doc.id)}
                            className="p-1.5 text-slate-600 hover:text-indigo-600 hover:bg-indigo-50 rounded-md transition-colors"
                            title="Download from MinIO"
                          >
                            <Download className="w-4 h-4" />
                          </a>
                          <button
                            onClick={() => handleDelete(doc.id, doc.filename)}
                            disabled={deletingId === doc.id}
                            className="p-1.5 text-slate-400 hover:text-rose-600 hover:bg-rose-50 rounded-md transition-colors disabled:opacity-50"
                            title="Delete Document"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

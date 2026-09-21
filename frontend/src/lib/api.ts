import {
  DocumentItem,
  DocumentUploadResponse,
  DocumentStatusResponse,
  ExtractionResult,
  ExtractedFieldsSummary,
  DocumentSearchResponse,
} from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Fetch list of all uploaded documents with pagination.
 */
export async function fetchDocuments(skip = 0, limit = 50): Promise<DocumentItem[]> {
  const res = await fetch(`${API_BASE_URL}/api/v1/documents/?skip=${skip}&limit=${limit}`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
    cache: "no-store",
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to fetch documents (${res.status})`);
  }

  return res.json();
}

/**
 * Fetch a single document by its ID.
 */
export async function fetchDocument(documentId: number): Promise<DocumentItem> {
  const res = await fetch(`${API_BASE_URL}/api/v1/documents/${documentId}`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
    cache: "no-store",
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to fetch document #${documentId} (${res.status})`);
  }

  return res.json();
}

/**
 * Fetch document processing status.
 */
export async function fetchDocumentStatus(documentId: number): Promise<DocumentStatusResponse> {
  const res = await fetch(`${API_BASE_URL}/api/v1/documents/${documentId}/status`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
    cache: "no-store",
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to fetch status for document #${documentId}`);
  }

  return res.json();
}

/**
 * Fetch document structured extraction results and validation details.
 */
export async function fetchDocumentResults(documentId: number): Promise<ExtractionResult | null> {
  const res = await fetch(`${API_BASE_URL}/api/v1/documents/${documentId}/results`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
    cache: "no-store",
  });

  if (res.status === 202) {
    return null; // Processing not complete yet
  }

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to fetch extraction results for document #${documentId}`);
  }

  return res.json();
}

/**
 * Fetch all extracted fields with coordinates and confidence scores for a document.
 */
export async function fetchDocumentFields(
  documentId: number,
  minConfidence?: number,
  fieldName?: string
): Promise<ExtractedFieldsSummary> {
  const params = new URLSearchParams();
  if (minConfidence !== undefined) params.append("min_confidence", minConfidence.toString());
  if (fieldName) params.append("field_name", fieldName);

  const query = params.toString() ? `?${params.toString()}` : "";
  const res = await fetch(`${API_BASE_URL}/api/v1/documents/${documentId}/fields${query}`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
    cache: "no-store",
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to fetch fields for document #${documentId}`);
  }

  return res.json();
}

/**
 * Search documents across filenames and extracted fields.
 */
export async function searchDocuments(
  query: string,
  fieldName?: string,
  status?: string,
  skip = 0,
  limit = 50
): Promise<DocumentSearchResponse> {
  const params = new URLSearchParams();
  params.append("q", query);
  if (fieldName) params.append("field_name", fieldName);
  if (status && status !== "ALL") params.append("status", status);
  params.append("skip", skip.toString());
  params.append("limit", limit.toString());

  const res = await fetch(`${API_BASE_URL}/api/v1/documents/search?${params.toString()}`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
    cache: "no-store",
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Search failed (${res.status})`);
  }

  return res.json();
}

/**
 * Upload a document file to backend API.
 */
export async function uploadDocumentFile(file: File, isHandwritten?: boolean): Promise<DocumentUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (isHandwritten !== undefined && isHandwritten !== null) {
    formData.append("is_handwritten", String(isHandwritten));
  }

  const res = await fetch(`${API_BASE_URL}/api/v1/documents/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to upload document (${res.status})`);
  }

  return res.json();
}

/**
 * Delete a document record and its storage.
 */
export async function deleteDocumentRecord(documentId: number): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/v1/documents/${documentId}`, {
    method: "DELETE",
  });

  if (!res.ok && res.status !== 204) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to delete document #${documentId}`);
  }
}

/**
 * Direct file download URL helper.
 */
export function getDirectDownloadUrl(documentId: number): string {
  return `${API_BASE_URL}/api/v1/documents/${documentId}/download`;
}

/**
 * Export URLs for Judge-Ready Digital Documents
 */
export function getPdfExportUrl(documentId: number): string {
  return `${API_BASE_URL}/api/v1/documents/${documentId}/export/pdf`;
}

export function getDocxExportUrl(documentId: number): string {
  return `${API_BASE_URL}/api/v1/documents/${documentId}/export/docx`;
}

export function getKannadaExportUrl(documentId: number): string {
  return `${API_BASE_URL}/api/v1/documents/${documentId}/export/kannada`;
}

export function getEnglishExportUrl(documentId: number): string {
  return `${API_BASE_URL}/api/v1/documents/${documentId}/export/english`;
}

/**
 * Interactive Translation API
 */
export async function translateText(
  text: string,
  sourceLang = "auto",
  targetLang = "en"
): Promise<{ original_text: string; translated_text: string; source_lang: string; target_lang: string }> {
  const res = await fetch(`${API_BASE_URL}/api/v1/documents/translate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text,
      source_lang: sourceLang,
      target_lang: targetLang,
    }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || "Translation request failed");
  }

  return res.json();
}

/**
 * Fetch human review items for a document.
 */
export async function fetchDocumentReviewItems(documentId: number) {
  const res = await fetch(`${API_BASE_URL}/api/v1/documents/${documentId}/review/items`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
    cache: "no-store",
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to fetch review items for document #${documentId}`);
  }

  return res.json();
}

/**
 * Submit a human review decision (ACCEPTED, CORRECTED, REJECTED)
 */
export async function submitDocumentReview(
  documentId: number,
  payload: {
    review_id: string;
    decision: string;
    corrected_text?: string | null;
    reviewer_notes?: string | null;
    reviewed_by?: string | null;
  }
) {
  const res = await fetch(`${API_BASE_URL}/api/v1/documents/${documentId}/review`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to submit review for document #${documentId}`);
  }

  return res.json();
}

export interface SaveBrowserOcrPayload {
  filename: string;
  file_hash: string;
  file_size?: number;
  text: string;
  confidence?: number;
  execution_provider?: string;
  latency_ms?: number;
  tokens?: number[];
  storage_path?: string;
  file_base64?: string;
}

/**
 * Save browser-executed TrOCR handwritten Kannada results to backend.
 */
export async function saveBrowserOcrResult(payload: SaveBrowserOcrPayload): Promise<DocumentUploadResponse> {
  const res = await fetch(`${API_BASE_URL}/api/v1/documents/browser-result`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to save browser OCR result (${res.status})`);
  }

  return res.json();
}



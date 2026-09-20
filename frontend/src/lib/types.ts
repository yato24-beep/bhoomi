export type DocumentStatus = "UPLOADED" | "PROCESSING" | "COMPLETED" | "FAILED" | string;

export interface BoundingBox {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
  unit?: string;
}

export interface DocumentItem {
  id: number;
  filename: string;
  file_hash: string;
  status: DocumentStatus;
  storage_path: string | null;
  created_at: string;
  updated_at?: string;
}

export interface ExtractedFieldItem {
  id: number;
  document_id: number;
  field_name: string;
  original_value: string | null;
  normalized_value: string | null;
  confidence_score: number;
  source_page: number;
  bounding_box: BoundingBox | null;
  english_value?: string | null;
  translation_status?: string | null;
  translation_engine?: string | null;
  source_type?: string | null;
}

export interface ExtractedFieldsSummary {
  document_id: number;
  total_fields: number;
  average_confidence: number;
  fields: ExtractedFieldItem[];
}

export interface ExtractionResult {
  id: number;
  document_id: number;
  extracted_data: Record<string, any>;
  confidence_score: number;
  is_valid: boolean;
  validation_info: Record<string, any>;
  processing_time_ms: number;
  created_at: string;
  recognition_confidence_raw?: number | null;
  calibrated_confidence?: number | null;
  stage_timings?: Record<string, number> | null;
}

export interface DocumentUploadResponse {
  message: string;
  is_duplicate: boolean;
  document: DocumentItem;
  task_id: string | null;
}

export interface DocumentStatusResponse {
  id: number;
  filename: string;
  status: DocumentStatus;
  storage_path: string | null;
}

export interface DocumentSearchItem extends DocumentItem {
  matched_fields?: ExtractedFieldItem[];
  match_source?: "filename" | "extracted_fields" | "both";
}

export interface DocumentSearchResponse {
  query: string;
  total_results: number;
  skip: number;
  limit: number;
  results: DocumentSearchItem[];
}

export interface ReviewItem {
  review_id: string;
  document_id: string;
  page_number: number;
  region_id: string;
  bbox?: BoundingBox | [number, number, number, number] | null;
  raw_ocr_text: string;
  recognizer: string;
  review_reason: string;
  recognizer_confidence_raw?: number | null;
  calibrated_confidence?: number | null;
  status: string;
  lifecycle_state: string;
  decision?: string | null;
  corrected_text?: string | null;
  reviewed_by?: string | null;
  reviewed_at?: string | null;
  reviewer_notes?: string | null;
  is_critical_field?: boolean;
  metadata?: Record<string, any>;
}

export interface ReviewCorrectionRequest {
  review_id: string;
  decision: "ACCEPTED" | "CORRECTED" | "REJECTED" | string;
  corrected_text?: string | null;
  reviewer_notes?: string | null;
  reviewed_by?: string | null;
}

export interface ReviewCorrectionResponse {
  status: string;
  review_id: string;
  document_id: number;
  item: ReviewItem;
  requires_human_review: boolean;
}


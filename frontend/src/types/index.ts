/**
 * TypeScript type definitions for PramanAI Frontend Interface.
 * Matches backend schemas and SSE streaming contracts.
 */

export interface CitationHighlight {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Citation {
  go_number: string;
  issuing_department: string;
  date: string;
  page_number: number;
  exact_text_excerpt: string;
  bounding_box_coordinates?: CitationHighlight | null;
}

export interface ConflictRecord {
  go_numbers: string[];
  description: string;
}

export interface OfficerContext {
  department: string;
  access_scope: string[];
}

export interface QueryFilters {
  department?: string | null;
  year_range?: [number, number] | number[] | null;
  policy_category?: string | null;
  go_number?: string | null;
}

export interface GraphStepState {
  node: string;
  step?: string;
  label: string;
  status: "started" | "completed" | "retrying";
}

export interface ApprovalActionPreview {
  trigger?: "low_confidence" | "conflict" | "personal_data";
  candidate_gos?: string[];
  confidence_score?: number;
  conflict_details?: string;
  description?: string;
  [key: string]: unknown;
}

export interface ApprovalRequiredData {
  checkpoint_id: string;
  graph_node: string;
  trigger: "low_confidence" | "conflict" | "personal_data";
  action_preview: ApprovalActionPreview;
}

export interface ChatMessage {
  id: string;
  role: "officer" | "agent" | "system";
  content: string;
  timestamp: string;
  userQuery?: string;
  queryFilters?: QueryFilters;
  confidence_score?: number;
  supersession_status?: string;
  citations?: Citation[];
  graceful_refusal?: boolean;
  isStreaming?: boolean;
  feedback?: {
    score?: boolean;
    comment?: string;
  };
}

export interface StateUpdateData {
  field: string;
  reducer: string;
  value: unknown;
  timestamp: string;
}

export interface ToolExecutionLog {
  toolName: string;
  toolCallId: string;
  state: string;
  input?: Record<string, unknown>;
  output?: Record<string, unknown>;
  timestamp: string;
}

export interface UserProfile {
  id: number;
  email: string;
  full_name: string;
  department: string;
  designation: string;
  role: string;
  created_at?: string | null;
}

export interface OfficerPersona {
  id: string;
  email: string;
  password: string;
  full_name: string;
  department: string;
  designation: string;
  role: string;
  description: string;
  badgeColor: string;
  badgeBg: string;
}

export interface ChatSessionItem {
  session_id: string;
  user_id?: number | null;
  title: string;
  department: string;
  created_at: string;
  updated_at: string;
}

export interface ChatSessionDetail {
  session_id: string;
  user_id?: number | null;
  title: string;
  department: string;
  created_at: string;
  updated_at: string;
  messages: ChatMessage[];
  citations: Citation[];
}


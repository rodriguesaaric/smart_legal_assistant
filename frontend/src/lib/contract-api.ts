import type { AnalysisResult, ChatMessage } from "@/types/contract";

const API_BASE = import.meta.env.VITE_API_URL || "";

/**
 * Analyze a contract — supports both raw text and file upload.
 */
export async function analyzeContract(
  input: { text: string; file?: undefined } | { file: File; text?: undefined },
  contractSource: string,
  userType: string,
  companyName?: string
): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append("contractSource", contractSource);
  formData.append("userType", userType);
  if (companyName) {
    formData.append("companyName", companyName);
  }

  if ("file" in input && input.file) {
    formData.append("file", input.file);
  } else if ("text" in input && input.text) {
    formData.append("contractText", input.text);
  }

  const resp = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: "Analysis failed" }));
    throw new Error(err.detail || err.error || `Error ${resp.status}`);
  }

  return resp.json();
}

/**
 * Ask a question about the analyzed contract via NLQ.
 */
export async function askContractQuestion(
  question: string,
  kClauses: number = 3,
  contractText?: string
): Promise<{
  question: string;
  expanded_query?: string;
  top_clauses: Array<{ id: string; text: string; similarity: number }>;
  answer: {
    summary: string;
    relevant_clauses: Array<{ id: number; excerpt: string }>;
    highlights: string[];
  };
}> {
  const resp = await fetch(`${API_BASE}/api/chat/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, k: kClauses, contractText }),
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: "Chat failed" }));
    throw new Error(err.detail || err.error || `Error ${resp.status}`);
  }

  return resp.json();
}

export interface RiskScore {
  ambiguity: number;
  maliciousIntent: number;
  clarity: number;
  compliance: number;
}

export interface RiskyPart {
  text: string;
  reason: string;
  severity: "low" | "medium" | "high";
  category: "ambiguity" | "malicious_intent" | "clarity" | "compliance";
  highlights?: string[];
}

export interface AnalysisResult {
  scores: RiskScore;
  overallRisk: "low" | "medium" | "high" | "critical";
  summary: string;
  documentAnalysis?: string;
  riskyParts: RiskyPart[];
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export type UserType = "consumer" | "company";

import { motion } from "framer-motion";
import type { RiskScore } from "@/types/contract";

interface RiskScoreDisplayProps {
  scores: RiskScore;
  overallRisk: string;
  summary: string;
  userType: string;
}

const scoreConfig = [
  { key: "ambiguity" as const, label: "Ambiguity", icon: "🔍", invertColor: true },
  { key: "maliciousIntent" as const, label: "Malicious Intent", icon: "⚠️", invertColor: true },
  { key: "clarity" as const, label: "Clarity", icon: "✨", invertColor: false },
  { key: "compliance" as const, label: "Compliance", icon: "📋", invertColor: false },
];

function getScoreColor(value: number, inverted: boolean): string {
  const v = inverted ? value : 100 - value;
  if (v <= 30) return "text-risk-low";
  if (v <= 60) return "text-risk-medium";
  if (v <= 80) return "text-risk-high";
  return "text-risk-critical";
}

function getBarColor(value: number, inverted: boolean): string {
  const v = inverted ? value : 100 - value;
  if (v <= 30) return "bg-risk-low";
  if (v <= 60) return "bg-risk-medium";
  if (v <= 80) return "bg-risk-high";
  return "bg-risk-critical";
}

function getOverallBadge(risk: string) {
  const map: Record<string, { bg: string; text: string }> = {
    low: { bg: "bg-risk-low/20 border-risk-low/40", text: "text-risk-low" },
    medium: { bg: "bg-risk-medium/20 border-risk-medium/40", text: "text-risk-medium" },
    high: { bg: "bg-risk-high/20 border-risk-high/40", text: "text-risk-high" },
    critical: { bg: "bg-risk-critical/20 border-risk-critical/40", text: "text-risk-critical" },
  };
  return map[risk] || map.medium;
}

const RiskScoreDisplay = ({ scores, overallRisk, summary, userType }: RiskScoreDisplayProps) => {
  const badge = getOverallBadge(overallRisk);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 }}
      className="space-y-5"
    >
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold font-mono tracking-tight text-foreground">
          Risk Analysis
        </h2>
        <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase border ${badge.bg} ${badge.text}`}>
          {overallRisk} risk
        </span>
      </div>

      <p className="text-sm text-muted-foreground leading-relaxed">{summary}</p>

      <div className="text-xs text-muted-foreground">
        Thresholds: <span className="text-primary font-medium">{userType === "company" ? "Company" : "Consumer"}</span> profile
      </div>

      <div className="space-y-4">
        {scoreConfig.map(({ key, label, icon, invertColor }, i) => (
          <motion.div
            key={key}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.15 + i * 0.08 }}
            className="space-y-1.5"
          >
            <div className="flex items-center justify-between">
              <span className="text-sm text-secondary-foreground flex items-center gap-1.5">
                {icon} {label}
              </span>
              <span className={`text-sm font-mono font-bold ${getScoreColor(scores[key], invertColor)}`}>
                {scores[key]}
              </span>
            </div>
            <div className="h-2 rounded-full bg-secondary overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${scores[key]}%` }}
                transition={{ duration: 0.8, delay: 0.2 + i * 0.1, ease: "easeOut" }}
                className={`h-full rounded-full ${getBarColor(scores[key], invertColor)}`}
              />
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
};

export default RiskScoreDisplay;

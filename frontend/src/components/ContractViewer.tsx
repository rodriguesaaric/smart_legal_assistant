import { motion } from "framer-motion";
import type { RiskyPart } from "@/types/contract";

interface ContractViewerProps {
  contractText: string;
  riskyParts: RiskyPart[];
}

const severityLabel: Record<string, string> = {
  high: "High Risk",
  medium: "Medium Risk",
  low: "Low Risk",
};

const severityClass: Record<string, string> = {
  high: "risk-highlight-high",
  medium: "risk-highlight-medium",
  low: "risk-highlight-low",
};

const categoryIcon: Record<string, string> = {
  ambiguity: "🔍",
  malicious_intent: "⚠️",
  clarity: "✨",
  compliance: "📋",
};

/**
 * Given a text and a list of trigger phrases, returns JSX with the
 * trigger phrases wrapped in bold + colored spans.
 */
const highlightTriggers = (
  text: string,
  highlights: string[],
  severity: string
): JSX.Element => {
  if (!highlights || highlights.length === 0) return <>{text}</>;

  // Deduplicate and sort by length descending (longest match first)
  const unique = [...new Set(highlights)].sort((a, b) => b.length - a.length);

  // Build a regex that matches any of the trigger phrases (case-insensitive)
  const escaped = unique.map((h) => h.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const regex = new RegExp(`(${escaped.join("|")})`, "gi");

  const parts = text.split(regex);

  const colorClass =
    severity === "high"
      ? "text-risk-high font-bold underline decoration-risk-high/50"
      : severity === "medium"
      ? "text-risk-medium font-bold underline decoration-risk-medium/50"
      : "text-risk-low font-bold";

  return (
    <>
      {parts.map((part, i) => {
        const isMatch = unique.some(
          (h) => h.toLowerCase() === part.toLowerCase()
        );
        return isMatch ? (
          <span key={i} className={colorClass}>
            {part}
          </span>
        ) : (
          <span key={i}>{part}</span>
        );
      })}
    </>
  );
};

const ContractViewer = ({ contractText, riskyParts }: ContractViewerProps) => {
  // Build highlighted text
  const buildHighlightedText = () => {
    if (!riskyParts.length) return <span>{contractText}</span>;

    // Normalize whitespace for matching (collapse spaces/newlines)
    const normalize = (s: string) => s.toLowerCase().replace(/\s+/g, " ").trim();
    const normalizedContract = normalize(contractText);

    // Build a position map: normalized index -> original index
    const normToOrigMap: number[] = [];
    let ni = 0;
    const contractLower = contractText.toLowerCase();
    let prevWasSpace = false;
    for (let oi = 0; oi < contractText.length; oi++) {
      const ch = contractText[oi];
      const isSpace = /\s/.test(ch);
      if (isSpace) {
        if (!prevWasSpace && ni > 0) {
          normToOrigMap[ni] = oi;
          ni++;
        }
        prevWasSpace = true;
      } else {
        prevWasSpace = false;
        normToOrigMap[ni] = oi;
        ni++;
      }
    }

    // Find all matches and sort by position
    const matches: { start: number; end: number; part: RiskyPart }[] = [];
    riskyParts.forEach((part) => {
      const normPart = normalize(part.text);

      // Try exact match first
      let idx = contractLower.indexOf(part.text.toLowerCase());

      // If not found, try normalized match
      if (idx === -1) {
        const normIdx = normalizedContract.indexOf(normPart);
        if (normIdx !== -1) {
          // Map back to original position
          const origStart = normToOrigMap[normIdx] ?? 0;
          // Find the end by searching for the last few words
          const lastWords = normPart.slice(-30);
          const endNormIdx = normalizedContract.indexOf(lastWords, normIdx);
          const origEnd = endNormIdx !== -1
            ? (normToOrigMap[endNormIdx + lastWords.length] ?? contractText.length)
            : Math.min(origStart + part.text.length + 50, contractText.length);
          matches.push({ start: origStart, end: origEnd, part });
          return;
        }

        // Last resort: match first 50 chars
        const snippet = normPart.slice(0, 50);
        const snipIdx = normalizedContract.indexOf(snippet);
        if (snipIdx !== -1) {
          const origStart = normToOrigMap[snipIdx] ?? 0;
          // Estimate end based on part text length ratio
          const endNormIdx = snipIdx + normPart.length;
          const origEnd = normToOrigMap[Math.min(endNormIdx, normToOrigMap.length - 1)] ?? contractText.length;
          matches.push({ start: origStart, end: origEnd, part });
          return;
        }
      } else {
        matches.push({ start: idx, end: idx + part.text.length, part });
      }
    });
    matches.sort((a, b) => a.start - b.start);

    // Remove overlaps
    const filtered: typeof matches = [];
    let lastEnd = 0;
    for (const m of matches) {
      if (m.start >= lastEnd) {
        filtered.push(m);
        lastEnd = m.end;
      }
    }

    const elements: JSX.Element[] = [];
    let cursor = 0;

    filtered.forEach((m, i) => {
      if (m.start > cursor) {
        elements.push(<span key={`t-${i}`}>{contractText.slice(cursor, m.start)}</span>);
      }

      // Highlighted clause block — with inner trigger highlights
      const clauseText = contractText.slice(m.start, m.end);
      elements.push(
        <span
          key={`h-${i}`}
          className={`risk-highlight ${severityClass[m.part.severity]} group relative`}
          title={`${severityLabel[m.part.severity]}: ${m.part.reason}`}
        >
          {m.part.highlights && m.part.highlights.length > 0
            ? highlightTriggers(clauseText, m.part.highlights, m.part.severity)
            : clauseText}
          <span className="hidden group-hover:block absolute z-10 bottom-full left-0 mb-2 p-2 rounded bg-card border border-border text-xs max-w-xs whitespace-normal shadow-lg">
            <span className="font-bold">{categoryIcon[m.part.category]} {severityLabel[m.part.severity]}</span>
            <br />
            {m.part.reason}
          </span>
        </span>
      );
      cursor = m.end;
    });

    if (cursor < contractText.length) {
      elements.push(<span key="end">{contractText.slice(cursor)}</span>);
    }

    return <>{elements}</>;
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
      className="space-y-4 flex flex-col h-full"
    >
      <div className="shrink-0 space-y-4">
          <h2 className="text-lg font-semibold font-mono tracking-tight text-foreground">
            📄 Contract Analysis
          </h2>

          <div className="flex gap-3 text-xs">
            <span className="flex items-center gap-1"><span className="w-3 h-1.5 rounded bg-risk-high inline-block" /> High</span>
            <span className="flex items-center gap-1"><span className="w-3 h-1.5 rounded bg-risk-medium inline-block" /> Medium</span>
            <span className="flex items-center gap-1"><span className="w-3 h-1.5 rounded bg-risk-low inline-block" /> Low</span>
          </div>
      </div>

      <div className="flex-1 min-h-[150px] p-4 rounded-lg bg-secondary border border-border font-mono text-sm leading-relaxed whitespace-pre-wrap overflow-y-auto custom-scrollbar">
        {buildHighlightedText()}
      </div>

      {riskyParts.length > 0 && (
        <div className="flex-1 min-h-[150px] flex flex-col space-y-2">
          <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider shrink-0">
            Flagged Clauses ({riskyParts.length})
          </h3>
          <div className="space-y-2 flex-1 overflow-y-auto pr-2 custom-scrollbar">
            {riskyParts.map((part, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 + i * 0.05 }}
                className={`p-3 rounded-md border text-xs ${
                  part.severity === "high"
                    ? "border-risk-high/30 bg-risk-high/5"
                    : part.severity === "medium"
                    ? "border-risk-medium/30 bg-risk-medium/5"
                    : "border-risk-low/30 bg-risk-low/5"
                }`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span>{categoryIcon[part.category]}</span>
                  <span className="font-bold uppercase">{severityLabel[part.severity]}</span>
                </div>
                <p className="text-muted-foreground italic">
                  "{part.highlights && part.highlights.length > 0
                    ? highlightTriggers(part.text, part.highlights, part.severity)
                    : part.text}"
                </p>
                <p className="mt-1 text-secondary-foreground">{part.reason}</p>
              </motion.div>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  );
};

export default ContractViewer;

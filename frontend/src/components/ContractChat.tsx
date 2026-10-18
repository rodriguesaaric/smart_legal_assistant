import { useState, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import { Send, MessageSquare } from "lucide-react";
import { Button } from "@/components/ui/button";
import { askContractQuestion } from "@/lib/contract-api";
import type { ChatMessage } from "@/types/contract";

interface ContractChatProps {
  contractText: string;
}

const ContractChat = ({ contractText }: ContractChatProps) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [kClauses, setKClauses] = useState(3);
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || isLoading) return;
    setInput("");

    const userMsg: ChatMessage = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const resp = await askContractQuestion(text, kClauses, contractText);

      // Build assistant response from the structured answer
      let answer = "";

      if (resp.answer?.summary) {
        answer += `**Answer:** ${resp.answer.summary}\n\n`;
      }

      if (resp.answer?.relevant_clauses?.length) {
        answer += `**Relevant Clauses (${resp.answer.relevant_clauses.length}):**\n`;
        resp.answer.relevant_clauses.forEach((c, i) => {
          answer += `${i + 1}. **Clause:** "${c.excerpt}"\n`;
        });
      }

      if (resp.top_clauses?.length) {
        answer += `\n---\n*Based on ${resp.top_clauses.length} matching clauses (top similarity: ${(resp.top_clauses[0]?.similarity * 100).toFixed(0)}%)*`;
      }

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: answer || "No relevant information found." },
      ]);
    } catch (e: any) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${e.message}` },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      className="flex flex-col h-full"
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-5 h-5 text-primary" />
          <h2 className="text-lg font-semibold font-mono tracking-tight text-foreground">
            Ask About Contract
          </h2>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs text-muted-foreground">Clauses (k):</label>
          <input
            type="number"
            min={1}
            max={10}
            value={kClauses}
            onChange={(e) => setKClauses(Math.max(1, Math.min(10, parseInt(e.target.value) || 1)))}
            className="w-14 px-2 py-1 rounded bg-secondary border border-border text-sm text-foreground text-center focus:outline-none focus:ring-1 focus:ring-primary"
          />
        </div>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-3 mb-3 min-h-[200px] max-h-[400px] pr-1">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
            Ask any question about your contract...
          </div>
        )}
        {messages.map((msg, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[85%] px-3 py-2 rounded-lg text-sm ${
                msg.role === "user"
                  ? "bg-primary text-primary-foreground"
                  : "bg-secondary text-secondary-foreground border border-border"
              }`}
            >
              {msg.role === "assistant" ? (
                <div className="prose prose-sm prose-invert max-w-none [&_p]:mb-2 [&_strong]:text-foreground [&_ol]:list-decimal [&_ol]:pl-4">
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                </div>
              ) : (
                <p>{msg.content}</p>
              )}
            </div>
          </motion.div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="px-3 py-2 rounded-lg bg-secondary border border-border">
              <span className="flex gap-1">
                <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                <span className="w-2 h-2 rounded-full bg-primary animate-pulse" style={{ animationDelay: "0.2s" }} />
                <span className="w-2 h-2 rounded-full bg-primary animate-pulse" style={{ animationDelay: "0.4s" }} />
              </span>
            </div>
          </div>
        )}
      </div>

      <div className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="Ask a question about the contract..."
          disabled={isLoading}
          className="flex-1 px-3 py-2 rounded-md bg-secondary border border-border text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-50"
        />
        <Button onClick={handleSend} disabled={isLoading || !input.trim()} size="icon">
          <Send className="w-4 h-4" />
        </Button>
      </div>
    </motion.div>
  );
};

export default ContractChat;

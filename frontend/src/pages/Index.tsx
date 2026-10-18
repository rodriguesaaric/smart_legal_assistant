import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Shield, Download } from "lucide-react";
import { toast } from "sonner";
import { generateFormalPDF } from "@/lib/pdf-export";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import ContractInput from "@/components/ContractInput";
import RiskScoreDisplay from "@/components/RiskScoreDisplay";
import ContractViewer from "@/components/ContractViewer";
import ContractChat from "@/components/ContractChat";
import { analyzeContract } from "@/lib/contract-api";
import type { AnalysisResult, UserType } from "@/types/contract";

const Index = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [contractText, setContractText] = useState("");
  const [userType, setUserType] = useState<UserType>("consumer");

  const handleAnalyze = async (
    input: { text: string } | { file: File },
    source: string,
    type: UserType,
    companyName: string
  ) => {
    setIsLoading(true);
    setUserType(type);

    // Attempt to extract text immediately if it's text
    if ("text" in input) {
      setContractText(input.text);
    } else {
      setContractText("Processing file...");
    }

    try {
      const analysis = await analyzeContract(input, source, type, companyName);

      // If we had a file upload, reconstruct the full text from clauses
      if ("file" in input && analysis.clauses) {
        const fullText = analysis.clauses
          .map((c: any) => c.text)
          .join("\n\n");
        setContractText(fullText);
      } else if ("file" in input && analysis.riskyParts?.length) {
        // Fallback: use risky parts text to show something
        const texts = analysis.riskyParts.map((rp: any) => rp.text);
        setContractText(texts.join("\n\n"));
      }

      setResult(analysis);
      toast.success("Contract analysis complete!");
    } catch (e: any) {
      toast.error(e.message || "Analysis failed");
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownloadPdf = () => {
    generateFormalPDF(result, contractText, userType);
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-3 flex items-center gap-3">
          <div className="p-1.5 rounded-lg bg-primary/10 glow-primary">
            <Shield className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h1 className="text-lg font-bold font-mono tracking-tight text-foreground">
              ContractShield
            </h1>
            <p className="text-xs text-muted-foreground">AI-Powered Contract Risk Analysis</p>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6">
        <AnimatePresence mode="wait">
          {!result ? (
            <motion.div
              key="input"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="max-w-2xl mx-auto"
            >
              <div className="rounded-xl border border-border bg-card p-6">
                <ContractInput onAnalyze={handleAnalyze} isLoading={isLoading} />
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="results"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="space-y-6"
            >
              {/* Action bar */}
              <div className="flex items-center justify-between">
                <h2 className="text-sm text-muted-foreground font-mono">Analysis Results</h2>
                <div className="flex gap-4 items-center">
                  <button
                    onClick={handleDownloadPdf}
                    className="text-xs flex items-center gap-1 px-3 py-1.5 rounded bg-primary text-primary-foreground hover:bg-primary/90 font-medium transition-colors"
                  >
                    <Download className="w-3 h-3" />
                    Download PDF Report
                  </button>
                  <button
                    onClick={() => { setResult(null); setContractText(""); }}
                    className="text-xs text-primary hover:underline font-medium"
                  >
                    ← New Analysis
                  </button>
                </div>
              </div>

              <div id="report-container" className="space-y-6 h-[calc(100vh-160px)] min-h-[700px] flex flex-col">
                {/* Print Title (Only visible in PDF if styles adjusted, or just render it always at top of #report-container) */}
                <div className="hidden print:block mb-4 shrink-0">
                  <h1 className="text-2xl font-bold">ContractShield Analysis Report</h1>
                  <p className="text-sm text-muted-foreground">Generated on: {new Date().toLocaleString()}</p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1 min-h-0">
                  {/* Left: Scores & Document Analysis */}
                  <div className="lg:col-span-4 rounded-xl border border-border bg-card p-5 flex flex-col h-full overflow-hidden shadow-lg">
                    <Tabs defaultValue="metrics" className="w-full flex-1 flex flex-col min-h-0">
                      <TabsList className="w-full shrink-0 mb-4 bg-secondary grid grid-cols-2">
                        <TabsTrigger value="metrics">Risk Metrics</TabsTrigger>
                        <TabsTrigger value="insights" disabled={!result.documentAnalysis}>
                          AI Summary
                        </TabsTrigger>
                      </TabsList>
                      
                      <TabsContent value="metrics" className="flex-1 overflow-y-auto pr-2 custom-scrollbar m-0 space-y-4">
                        <RiskScoreDisplay
                          scores={result.scores}
                          overallRisk={result.overallRisk}
                          summary={result.summary}
                          userType={userType}
                        />
                      </TabsContent>
                      
                      {result.documentAnalysis && (
                        <TabsContent value="insights" className="flex-1 overflow-y-auto pr-2 custom-scrollbar m-0">
                          <div className="prose prose-sm prose-invert max-w-none text-muted-foreground prose-headings:text-primary prose-headings:border-b prose-headings:border-primary/20 prose-headings:pb-1 prose-strong:text-amber-400 prose-strong:font-semibold prose-p:leading-relaxed prose-li:marker:text-primary prose-table:w-full prose-th:bg-secondary prose-th:p-2 prose-td:p-2 prose-td:border-t prose-td:border-border prose-table:border prose-table:border-border">
                            <ReactMarkdown 
                              remarkPlugins={[remarkGfm]}
                              components={{
                                p: ({ node, children, ...props }) => {
                                  // Colorize standalone paragraphs that act as headings
                                  const text = String(children);
                                  if (text.trim() === "Pros" || text.trim() === "Advantages") {
                                    return <p className="text-green-500 font-bold uppercase tracking-wider mt-4" {...props}>{children}</p>;
                                  }
                                  if (text.trim() === "Cons" || text.trim() === "Disadvantages") {
                                    return <p className="text-red-500 font-bold uppercase tracking-wider mt-4" {...props}>{children}</p>;
                                  }
                                  return <p {...props}>{children}</p>;
                                },
                                strong: ({ node, children, ...props }) => {
                                  // Colorize inline bold texts that act as headings
                                  const text = String(children);
                                  if (text.trim() === "Pros" || text.trim() === "Advantages") {
                                    return <strong className="text-green-500 font-bold uppercase tracking-wider block mt-4" {...props}>{children}</strong>;
                                  }
                                  if (text.trim() === "Cons" || text.trim() === "Disadvantages") {
                                    return <strong className="text-red-500 font-bold uppercase tracking-wider block mt-4" {...props}>{children}</strong>;
                                  }
                                  // Sometimes it prints "Loss of Content..." bold. Leave default amber.
                                  return <strong {...props}>{children}</strong>;
                                },
                                h3: ({ node, children, ...props }) => {
                                  // In case the AI uses markdown H3 ### for Pros/Cons
                                  const text = String(children);
                                  if (text.trim() === "Pros" || text.trim() === "Advantages") {
                                    return <h3 className="text-green-500 font-bold uppercase tracking-wider mt-4 border-b-0 pb-0" {...props}>{children}</h3>;
                                  }
                                  if (text.trim() === "Cons" || text.trim() === "Disadvantages") {
                                    return <h3 className="text-red-500 font-bold uppercase tracking-wider mt-4 border-b-0 pb-0" {...props}>{children}</h3>;
                                  }
                                  return <h3 {...props}>{children}</h3>;
                                }
                              }}
                            >
                              {result.documentAnalysis}
                            </ReactMarkdown>
                          </div>
                        </TabsContent>
                      )}
                    </Tabs>

                    {/* Display analysis timings at the very bottom */}
                    {(result as any).metrics?.timing && (
                      <div className="mt-4 pt-4 border-t border-border flex flex-col gap-1 text-[10px] text-muted-foreground shrink-0">
                         <div className="flex justify-between">
                            <span>Random Forest Clause Analysis:</span>
                            <span>{((result as any).metrics.timing.total_ms / 1000).toFixed(2)}s</span>
                         </div>
                         <div className="flex justify-between">
                            <span>LLM Document Analysis:</span>
                            <span>{((result as any).metrics.timing.document_analysis_ms / 1000).toFixed(2)}s</span>
                         </div>
                      </div>
                    )}
                  </div>

                  {/* Center: Contract viewer */}
                  <div className="lg:col-span-4 rounded-xl border border-border bg-card p-5 h-full overflow-hidden shadow-lg">
                    <ContractViewer
                      contractText={contractText}
                      riskyParts={result.riskyParts}
                    />
                  </div>

                  {/* Right: Chat */}
                  <div className="lg:col-span-4 rounded-xl border border-border bg-card p-5 h-full overflow-hidden shadow-lg">
                    <ContractChat contractText={contractText} />
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
};

export default Index;

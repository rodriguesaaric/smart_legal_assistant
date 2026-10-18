import { useState, useRef } from "react";
import { motion } from "framer-motion";
import { Shield, FileText, Upload, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { UserType } from "@/types/contract";

interface ContractInputProps {
  onAnalyze: (
    input: { text: string } | { file: File },
    source: string,
    userType: UserType,
    companyName: string
  ) => void;
  isLoading: boolean;
}

const ACCEPTED_TYPES = [
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "text/plain",
];
const ACCEPTED_EXT = ".pdf,.docx,.txt";

const ContractInput = ({ onAnalyze, isLoading }: ContractInputProps) => {
  const [contractText, setContractText] = useState("");
  const [contractSource, setContractSource] = useState("");
  const [userType, setUserType] = useState<UserType>("consumer");
  const [companyName, setCompanyName] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleSubmit = () => {
    if (selectedFile) {
      onAnalyze({ file: selectedFile }, contractSource, userType, companyName);
    } else if (contractText.trim()) {
      onAnalyze({ text: contractText }, contractSource, userType, companyName);
    }
  };

  const handleFileDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files[0];
    if (file) setSelectedFile(file);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setSelectedFile(file);
  };

  const clearFile = () => {
    setSelectedFile(null);
    if (fileRef.current) fileRef.current.value = "";
  };

  const canSubmit = (selectedFile || contractText.trim()) && !isLoading;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-5"
    >
      <div className="flex items-center gap-3 mb-2">
        <div className="p-2 rounded-lg bg-primary/10 glow-primary">
          <Shield className="w-5 h-5 text-primary" />
        </div>
        <h2 className="text-lg font-semibold font-mono tracking-tight text-foreground">
          Contract Input
        </h2>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            Contract Source
          </label>
          <input
            type="text"
            value={contractSource}
            onChange={(e) => setContractSource(e.target.value)}
            placeholder="e.g. Sponsorship Agreement"
            className="w-full px-3 py-2 rounded-md bg-secondary border border-border text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
          />
        </div>
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            User Type
          </label>
          <div className="flex gap-2">
            {(["consumer", "company"] as UserType[]).map((type) => (
              <button
                key={type}
                onClick={() => setUserType(type)}
                className={`flex-1 px-3 py-2 rounded-md text-sm font-medium transition-all ${userType === type
                    ? "bg-primary text-primary-foreground glow-primary"
                    : "bg-secondary text-secondary-foreground hover:bg-secondary/80"
                  }`}
              >
                {type === "consumer" ? "👤 Consumer" : "🏢 Company"}
              </button>
            ))}
          </div>
        </div>
      </div>

      {userType === "company" && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          className="space-y-1.5"
        >
          <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            Organization / Party Name
          </label>
          <input
            type="text"
            value={companyName}
            onChange={(e) => setCompanyName(e.target.value)}
            placeholder="e.g. Organization Name"
            className="w-full px-3 py-2 rounded-md bg-secondary border border-border text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
          />
        </motion.div>
      )}

      {/* File upload area */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          Upload Contract File
        </label>
        <div
          onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleFileDrop}
          onClick={() => fileRef.current?.click()}
          className={`relative flex flex-col items-center justify-center p-6 rounded-lg border-2 border-dashed cursor-pointer transition-all ${dragActive
              ? "border-primary bg-primary/5"
              : selectedFile
                ? "border-primary/50 bg-primary/5"
                : "border-border hover:border-primary/40 bg-secondary/50"
            }`}
        >
          <input
            ref={fileRef}
            type="file"
            accept={ACCEPTED_EXT}
            onChange={handleFileSelect}
            className="hidden"
          />
          {selectedFile ? (
            <div className="flex items-center gap-2 text-sm text-foreground">
              <FileText className="w-5 h-5 text-primary" />
              <span className="font-medium">{selectedFile.name}</span>
              <span className="text-muted-foreground">
                ({(selectedFile.size / 1024).toFixed(1)} KB)
              </span>
              <button
                onClick={(e) => { e.stopPropagation(); clearFile(); }}
                className="p-1 rounded hover:bg-secondary"
              >
                <X className="w-4 h-4 text-muted-foreground" />
              </button>
            </div>
          ) : (
            <>
              <Upload className="w-8 h-8 text-muted-foreground mb-2" />
              <p className="text-sm text-muted-foreground">
                Drop a file here or <span className="text-primary font-medium">click to browse</span>
              </p>
              <p className="text-xs text-muted-foreground mt-1">PDF, DOCX, or TXT</p>
            </>
          )}
        </div>
      </div>

      {/* Divider */}
      <div className="flex items-center gap-3">
        <div className="flex-1 h-px bg-border" />
        <span className="text-xs text-muted-foreground uppercase tracking-wider">or paste text</span>
        <div className="flex-1 h-px bg-border" />
      </div>

      {/* Text input */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          Contract Text
        </label>
        <textarea
          value={contractText}
          onChange={(e) => setContractText(e.target.value)}
          placeholder="Paste your contract text here..."
          rows={10}
          disabled={!!selectedFile}
          className="w-full px-3 py-2 rounded-md bg-secondary border border-border text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary font-mono leading-relaxed resize-none disabled:opacity-40"
        />
      </div>

      <Button
        onClick={handleSubmit}
        disabled={!canSubmit}
        className="w-full glow-primary"
      >
        {isLoading ? (
          <span className="flex items-center gap-2">
            <span className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
            Analyzing...
          </span>
        ) : (
          <span className="flex items-center gap-2">
            <FileText className="w-4 h-4" />
            Analyze Contract
          </span>
        )}
      </Button>
    </motion.div>
  );
};

export default ContractInput;

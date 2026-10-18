import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";
import type { AnalysisResult, UserType } from "@/types/contract";

export const generateFormalPDF = (
  result: AnalysisResult | null,
  contractText: string,
  userType: UserType
) => {
  if (!result) return;

  // Sanitize text to remove non-ANSI characters that jsPDF Helvetica can't render
  const sanitizeForPDF = (text: string): string => {
    if (!text) return "";
    return text
        .replace(/₹/g, "Rs. ")
        .replace(/[\u2018\u2019]/g, "'")
        .replace(/[\u201C\u201D]/g, '"')
        .replace(/[\u2013\u2014]/g, "-")
        .replace(/…/g, "...")
        .replace(/™/g, "(TM)")
        .replace(/®/g, "(R)")
        .replace(/©/g, "(C)");
  };

  const doc = new jsPDF({
    orientation: "portrait",
    unit: "mm",
    format: "a4",
  });

  const pageWidth = doc.internal.pageSize.getWidth();
  
  // 1. Cover / Header
  doc.setFontSize(22);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(33, 37, 41);
  doc.text("ContractShield Risk Assessment", 14, 22);
  
  doc.setFontSize(10);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(108, 117, 125);
  doc.text(`Report Generated: ${new Date().toLocaleString()}`, 14, 30);
  doc.text(`Analysis Mode: ${userType === "consumer" ? "Consumer Protection" : "Corporate Negotiation"}`, 14, 36);
  doc.text(`Overall Risk Rating: ${result.overallRisk.toUpperCase()}`, 14, 42);

  // 2. Summary Scores Table
  doc.setFontSize(14);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(33, 37, 41);
  doc.text("Core Metrics", 14, 55);

  autoTable(doc, {
    startY: 60,
    head: [["Category", "Score", "Rating"]],
    body: [
      ["Ambiguity", `${result.scores.ambiguity}/100`, getRatingLabel(result.scores.ambiguity, true)],
      ["Malicious Intent", `${result.scores.maliciousIntent}/100`, getRatingLabel(result.scores.maliciousIntent, true)],
      ["Clarity", `${result.scores.clarity}/100`, getRatingLabel(result.scores.clarity, false)],
      ["Compliance", `${result.scores.compliance}/100`, getRatingLabel(result.scores.compliance, false)],
    ],
    theme: "grid",
    headStyles: { fillColor: [63, 131, 248] }, // Primary blue
    styles: { fontSize: 10, cellPadding: 3 },
  });

  // 3. AI Insights (Document Analysis)
  let nextY = (doc as any).lastAutoTable.finalY + 15;
  
  if (result.documentAnalysis) {
    if (nextY > 250) {
      doc.addPage();
      nextY = 20;
    }
    
    doc.setFontSize(14);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(33, 37, 41);
    doc.text(userType === "consumer" ? "Document Pros & Cons (AI Summary)" : "Negotiation Suggestions (AI Summary)", 14, nextY);
    
    // Increment Y after the header title to prevent overlapping!
    nextY += 8;

    doc.setFontSize(10);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(52, 58, 64);
    
    // Parse Markdown tables vs text and render dynamically
    const lines = sanitizeForPDF(result.documentAnalysis).split('\n');
    let tableRows: string[][] = [];
    let textBlock: string[] = [];

    const printRichText = (text: string, startX: number, startY: number, maxWidth: number) => {
        const parts = text.split('**');
        let currentX = startX;
        let currentY = startY;
        let isBoldLine = false;

        for (let i = 0; i < parts.length; i++) {
            let chunk = parts[i];
            if (!chunk) {
                isBoldLine = !isBoldLine;
                continue;
            }

            // Clean other symbols
            chunk = chunk.replace(/\*/g, '·').replace(/#/g, '').replace(/[\t]+/g, ' ');
            
            doc.setFont("helvetica", isBoldLine ? "bold" : "normal");
            
            // Split by whitespace but keep the whitespace for measurement
            const words = chunk.split(/(\s+)/);
            
            for (const word of words) {
                if (!word) continue;
                const wWidth = doc.getTextWidth(word);
                
                // Wrap to next line if exceeds width
                if (currentX + wWidth > startX + maxWidth && word.trim() !== "") {
                    currentX = startX + 4; // slight indent for wrapped lines
                    currentY += 5;
                    // Pagination check
                    if (currentY > 280) {
                        doc.addPage();
                        currentY = 20;
                    }
                }
                
                // Skip rendering space if it's the exact start of a line
                if (currentX === startX && word.trim() === "") continue;
                
                doc.text(word, currentX, currentY);
                currentX += wWidth;
            }
            
            isBoldLine = !isBoldLine;
        }
        return currentY;
    };

    const flushTextBlock = () => {
      if (textBlock.length > 0) {
        for (const line of textBlock) {
            let cleanLine = line.trim();
            
            // Detect full-line headings
            if (cleanLine.startsWith('#')) {
                cleanLine = `**${cleanLine.replace(/^#+\s*/, '')}**`;
            }
            
            if (cleanLine === "") {
                nextY += 2;
                continue;
            }
            
            // Render rich text line
            nextY = printRichText(cleanLine, 14, nextY, pageWidth - 28);
            nextY += 5;
        }
        
        doc.setFont("helvetica", "normal");
        textBlock = [];
        nextY += 3;
      }
    };

    const flushTableBlock = () => {
      if (tableRows.length > 0) {
        if (nextY > 250) {
            doc.addPage();
            nextY = 20;
        }
        autoTable(doc, {
          startY: nextY,
          head: tableRows.length > 0 ? [tableRows[0]] : [],
          body: tableRows.length > 1 ? tableRows.slice(1) : [],
          theme: "grid",
          styles: { fontSize: 9, cellPadding: 3, overflow: 'linebreak' },
          headStyles: { fillColor: [108, 117, 125] },
        });
        nextY = (doc as any).lastAutoTable.finalY + 10;
        tableRows = [];
      }
    };

    for (let line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
        flushTextBlock();
        if (trimmed.replace(/ /g, '').match(/^\|[-|:]+\|$/)) {
          // Markdown syntax separator row, skip it
          continue;
        }
        const cells = trimmed.split('|').slice(1, -1).map(c => c.trim().replace(/\*\*/g, "").replace(/#/g, ""));
        tableRows.push(cells);
      } else {
        flushTableBlock();
        textBlock.push(line);
      }
    }
    
    flushTextBlock();
    flushTableBlock();
    
    nextY += 5;
  }

  // 4. High-Risk Clauses Table
  if (result.riskyParts && result.riskyParts.length > 0) {
    if (nextY > 230) {
      doc.addPage();
      nextY = 20;
    }

    doc.setFontSize(14);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(33, 37, 41);
    doc.text("Flagged Clauses & Issues", 14, nextY);
    
    const tableBody = result.riskyParts.map(part => [
       (part.severity || "UNKNOWN").toUpperCase(),
       (part.category || "").replace(/_/g, " ").toUpperCase(),
       sanitizeForPDF(part.text)
    ]);

    autoTable(doc, {
      startY: nextY + 5,
      head: [["Severity", "Type", "Extracted Clause Text"]],
      body: tableBody,
      theme: "grid",
      columnStyles: {
        0: { cellWidth: 22, fontStyle: 'bold' },
        1: { cellWidth: 35 },
        2: { cellWidth: "auto" }
      },
      headStyles: { fillColor: [224, 36, 36] }, // Reddish
      styles: { fontSize: 9, cellPadding: 3, overflow: 'linebreak' },
      didParseCell: function(data: any) {
        if (data.column.index === 0 && data.cell.section === 'body') {
            if (data.cell.raw === 'HIGH' || data.cell.raw === 'CRITICAL') {
                data.cell.styles.textColor = [224, 36, 36]; // Red text for high risk
            } else if (data.cell.raw === 'MEDIUM') {
                data.cell.styles.textColor = [202, 138, 4]; // Yellow text
            } else {
                data.cell.styles.textColor = [22, 163, 74]; // Green text
            }
        }
      }
    });
  }

  // Footer for all pages
  const pageCount = (doc.internal as any).getNumberOfPages();
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    doc.setFontSize(8);
    doc.setTextColor(150);
    doc.text(
      `Page ${i} of ${pageCount} - ContractShield AI Assessment`,
      pageWidth / 2,
      doc.internal.pageSize.getHeight() - 10,
      { align: "center" }
    );
  }

  // Save the PDF
  doc.save(`ContractShield_Report_${new Date().toISOString().split("T")[0]}.pdf`);
};

function getRatingLabel(score: number, inverted: boolean): string {
  const v = inverted ? score : 100 - score;
  if (v <= 30) return "Low Risk / Favorable";
  if (v <= 60) return "Moderate / Review";
  return "High Risk / Unfavorable";
}

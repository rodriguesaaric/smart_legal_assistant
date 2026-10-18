"""
Generate synthetic training data for the clause boundary detection model.

Each sample has:
  - "text": a concatenated legal document snippet
  - "clause_starts": list of character offsets where each clause begins
"""

import json
import random

CLAUSE_TEMPLATES = [
    # Definitions
    [
        "1. Definitions\nIn this Agreement, unless the context otherwise requires, the following terms shall have the meanings assigned to them below.",
        "1.1 \"Agreement\" means this contract including all schedules and annexures attached hereto.",
        "1.2 \"Confidential Information\" means all non-public information disclosed by either party to the other, whether orally, in writing, or by inspection.",
        "1.3 \"Effective Date\" means the date on which this Agreement is executed by both parties.",
        "1.4 \"Services\" means the professional services to be provided by the Service Provider as described in Schedule A.",
        "1.5 \"Term\" means the period commencing on the Effective Date and continuing until terminated in accordance with this Agreement.",
    ],
    # Obligations
    [
        "2. Obligations of the Service Provider\nThe Service Provider shall perform the Services with all due skill, care and diligence.",
        "2.1 The Service Provider shall comply with all applicable laws, regulations and industry standards in the performance of the Services.",
        "2.2 The Service Provider shall maintain adequate insurance coverage for the duration of this Agreement.",
        "2.3 The Service Provider shall provide monthly progress reports to the Client detailing the status of deliverables.",
    ],
    # Payment terms
    [
        "3. Payment Terms\nThe Client shall pay the Service Provider the fees set out in Schedule B.",
        "3.1 All invoices shall be payable within thirty (30) days from the date of receipt.",
        "3.2 Late payments shall incur interest at the rate of 1.5 percent per month on the outstanding amount.",
        "3.3 The Client shall reimburse the Service Provider for all reasonable out-of-pocket expenses incurred in connection with the Services.",
    ],
    # Termination
    [
        "4. Termination\nEither party may terminate this Agreement by providing sixty (60) days prior written notice to the other party.",
        "4.1 This Agreement may be terminated immediately by either party if the other party commits a material breach that remains uncured for thirty (30) days after written notice.",
        "4.2 Upon termination, the Service Provider shall deliver all work product completed up to the date of termination.",
        "4.3 Termination shall not affect any accrued rights or obligations of either party.",
    ],
    # Confidentiality
    [
        "5. Confidentiality\nEach party agrees to keep confidential all Confidential Information received from the other party.",
        "5.1 The receiving party shall not disclose Confidential Information to any third party without the prior written consent of the disclosing party.",
        "5.2 The obligations of confidentiality shall survive the termination of this Agreement for a period of five (5) years.",
        "5.3 Confidential Information shall not include information that is or becomes publicly available through no fault of the receiving party.",
    ],
    # Liability
    [
        "6. Limitation of Liability\nNeither party shall be liable for any indirect, incidental, special, consequential or punitive damages.",
        "6.1 The total aggregate liability of either party under this Agreement shall not exceed the total fees paid or payable under this Agreement.",
        "6.2 Nothing in this Agreement shall limit liability for death or personal injury caused by negligence, fraud or fraudulent misrepresentation.",
    ],
    # IP Rights
    [
        "7. Intellectual Property\nAll intellectual property rights in the deliverables shall vest in the Client upon full payment.",
        "7.1 The Service Provider grants the Client a perpetual, irrevocable, royalty-free license to use any pre-existing intellectual property incorporated in the deliverables.",
        "7.2 The Service Provider warrants that the deliverables do not infringe any third-party intellectual property rights.",
    ],
    # Indemnification
    [
        "8. Indemnification\nThe Service Provider shall indemnify and hold harmless the Client from any claims, losses, damages or expenses arising from the Service Provider's breach of this Agreement.",
        "8.1 The Client shall promptly notify the Service Provider of any claim for which indemnification is sought.",
        "8.2 The indemnifying party shall have the right to control the defense of any such claim.",
    ],
    # Force Majeure
    [
        "9. Force Majeure\nNeither party shall be liable for any failure or delay in performance due to circumstances beyond its reasonable control.",
        "9.1 Events of force majeure include but are not limited to natural disasters, wars, epidemics, government orders, and acts of terrorism.",
        "9.2 The affected party shall promptly notify the other party of the occurrence and expected duration of the force majeure event.",
    ],
    # Governing Law
    [
        "10. Governing Law and Dispute Resolution\nThis Agreement shall be governed by and construed in accordance with the laws of India.",
        "10.1 Any dispute arising out of or in connection with this Agreement shall first be attempted to be resolved through mediation.",
        "10.2 If mediation fails, the dispute shall be referred to binding arbitration in accordance with the Arbitration and Conciliation Act, 1996.",
        "10.3 The seat of arbitration shall be Mumbai, India, and the language of arbitration shall be English.",
    ],
    # Non-compete
    [
        "11. Non-Competition\nDuring the term of this Agreement and for a period of twelve (12) months thereafter, the Service Provider shall not directly or indirectly engage in any business that competes with the Client.",
        "11.1 The Service Provider acknowledges that the restrictions in this clause are reasonable and necessary for the protection of the Client's legitimate business interests.",
    ],
    # Data Protection
    [
        "12. Data Protection\nEach party shall comply with all applicable data protection and privacy laws in relation to any personal data processed under this Agreement.",
        "12.1 The Service Provider shall implement appropriate technical and organizational measures to protect personal data against unauthorized access, loss or destruction.",
        "12.2 In the event of a data breach, the Service Provider shall notify the Client within 72 hours of becoming aware of the breach.",
    ],
    # Notices
    [
        "13. Notices\nAll notices under this Agreement shall be in writing and shall be deemed given when delivered personally, sent by registered post, or transmitted by email.",
        "13.1 Notices to the Client shall be sent to the address specified in Schedule C.",
        "13.2 Either party may change its notice address by providing written notice to the other party.",
    ],
    # General provisions
    [
        "14. General Provisions\nThis Agreement constitutes the entire agreement between the parties and supersedes all prior negotiations, representations and agreements.",
        "14.1 No amendment to this Agreement shall be effective unless made in writing and signed by both parties.",
        "14.2 The failure of either party to enforce any provision of this Agreement shall not constitute a waiver of such provision.",
        "14.3 If any provision of this Agreement is held invalid or unenforceable, the remaining provisions shall continue in full force and effect.",
        "14.4 This Agreement may not be assigned by either party without the prior written consent of the other party.",
    ],
    # Sponsorship specific
    [
        "The Brand shall be promoted as the Title Sponsor of the Event organized by the Organising Committee at the Venue on the official Event Dates.",
        "The Brand's logo shall be incorporated into the Integrated Logo Unit of the Festival and shall be included in all official communications.",
        "All Event-related communication shall carry the mention of the Brand name, followed by the agreed sponsor hierarchy.",
        "The Organising Committee shall ensure that the festival will not have any association with any competing platform, application, or partners.",
    ],
    # Exclusivity
    [
        "The Brand shall retain the right to approve the final artist lineup for the festival to ensure alignment with its expectations.",
        "No other sponsor's audio-visual content shall be displayed at the Venue including on LED screens and no sponsor logos or branding shall appear on-screen at any time.",
        "There shall be no branding on the three audience-facing sides of the venue by any third-party sponsor, festival, or college.",
        "The stage shall be exclusively branded by the Brand, with no branding permitted by any third party.",
    ],
    # Registration
    [
        "The QR code for the Brand's official app shall be a mandatory entry requirement to the Sponsored Night.",
        "The Organising Committee shall ensure a minimum of 3,000 successful registrations through the Brand's registration platform.",
        "These numbers will be validated before the Sponsored Night and failure to meet the target may result in penalties.",
    ],
]

def generate_document(num_sections=None):
    """Generate a random legal document with clause boundary annotations."""
    if num_sections is None:
        num_sections = random.randint(3, 8)
    
    selected = random.sample(CLAUSE_TEMPLATES, min(num_sections, len(CLAUSE_TEMPLATES)))
    
    all_clauses = []
    for section in selected:
        # Randomly pick some clauses from each section
        n = random.randint(1, len(section))
        chosen = random.sample(section, n)
        all_clauses.extend(chosen)
    
    # Shuffle to create realistic variation
    if random.random() > 0.5:
        random.shuffle(all_clauses)
    
    # Join with separators
    separators = ["\n\n", "\n", " ", "\n\n\n"]
    text = ""
    clause_starts = []
    
    for i, clause in enumerate(all_clauses):
        clause_starts.append(len(text))
        text += clause
        if i < len(all_clauses) - 1:
            text += random.choice(separators)
    
    return {"text": text, "clause_starts": clause_starts}


def main():
    random.seed(42)
    
    documents = []
    for _ in range(60):
        doc = generate_document()
        documents.append(doc)
    
    with open("training/train.json", "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(documents)} training documents")
    print(f"Sample doc has {len(documents[0]['clause_starts'])} clauses")
    print(f"Saved to training/train.json")


if __name__ == "__main__":
    main()

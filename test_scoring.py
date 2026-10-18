"""Test with user's exact Consumer + Company clauses."""
from scoring.analyzer import analyze_clauses

# ── CONSUMER CLAUSES ──
consumer_clauses = [
    {"id": 1,  "text": "The Provider agrees to offer access to its web-based platform for managing digital content and communication tools. Services will be available 24/7 except during scheduled maintenance."},
    {"id": 2,  "text": "The Consumer agrees to pay a monthly subscription fee of Rs.499. Payments are billed automatically and can be canceled anytime before the next billing cycle."},
    {"id": 3,  "text": "The Provider will collect and process user data in accordance with applicable data protection laws and will not sell personal data to third parties without explicit consent."},
    {"id": 4,  "text": "The Provider shall not be liable for any indirect, incidental, or consequential damages arising from the use of the service, including loss of data or profits."},
    {"id": 5,  "text": "The Provider does not guarantee uninterrupted access and may suspend services temporarily for maintenance, upgrades, or unforeseen issues without prior notice."},
    {"id": 6,  "text": "The Provider reserves the right to suspend or terminate accounts that violate terms of use or engage in suspicious activity, at its sole discretion."},
    {"id": 7,  "text": "The Provider may modify this Agreement at any time without notifying the Consumer. Continued use of the service constitutes acceptance of the revised terms."},
    {"id": 8,  "text": "All content uploaded by the Consumer becomes the property of the Provider, including rights to use, reproduce, and distribute such content without compensation."},
    {"id": 9,  "text": "The Consumer agrees to resolve all disputes through binding arbitration and waives any right to participate in class-action lawsuits or seek remedies through courts."},
    {"id": 10, "text": "Subscriptions will automatically renew for a minimum of 12 months, and cancellation is only permitted within a 24-hour window before renewal."},
]

# ── COMPANY CLAUSES ──
company_clauses = [
    {"id": 1,  "text": "The Brand shall be promoted as the Title Sponsor of the Event organized by the Organising Committee at the Venue on the official Event Dates."},
    {"id": 2,  "text": "The Brand's logo shall be incorporated into the Integrated Logo Unit of the Festival and shall be included in all official communications - both online and offline, including digital banners, social media posts, websites, merchandise, promotional material, newspaper, Radio, official invites, stage branding, and press coverage."},
    {"id": 3,  "text": "All Event-related communication shall carry the mention: Brand presents Incredia 2026, followed by: In association with or Co-powered by any other sponsors, as per the agreed sponsor hierarchy. This format shall be used consistently across all online, offline, and media-facing communication touchpoints."},
    {"id": 4,  "text": "The Organising Committee shall ensure that the festival will not have any association, neither offline nor online with any other platform, applications, partners, or their affiliates that operate in the same or parallel space or business. A detailed list will also be included in the MOU."},
    {"id": 5,  "text": "The festival will ensure the following dedicated social media collateral across all the official handles of the festival. 8 Static Posts before during and after the festival. 20 stories Including before during and after the festival. There shall be no direct promotional posts about the Brand; all content shall focus solely on event-related activities, engagement, and festival-centric communication."},
    {"id": 6,  "text": "All social media content and releases related to the Sponsored Pronight including but not limited to artist announcements, teaser posts, artist bytes, countdowns, and engagement content shall be exclusively created and provided by the Brand. These shall carry only the Festival and Host College logos; no third-party or sponsor logos will be permitted. Any additional content created by the Organising Committee will require prior written approval from the Brand. The Brand reserves the right to disallow such additional content if not aligned with its brand guidelines or objectives."},
    {"id": 7,  "text": "The Brand's logo shall be featured on the Festival's official sponsors page, along with a hyperlink to the Brand's official website."},
    {"id": 8,  "text": "All attendees of the Event including students from the host institution shall be required to register through the Brand's official app or the event-specific landing page, as a mandatory entry requirement to the Sponsored Night. The QR code for this registration flow shall be provided by the Brand and must be prominently displayed across all communications related to the Sponsored Night, including but not limited to event-related posters, social media posts, and any other student-facing communication. In addition, the Organising Committee shall ensure a minimum of 3,000 successful registrations through the Brand's registration platform. These numbers will be validated before the Sponsored Night."},
    {"id": 9,  "text": "The stage shall be exclusively branded by the Brand, with no branding permitted by any third party including other sponsors, college branding but excluding festival branding."},
    {"id": 10, "text": "No other sponsor's audio-visual content shall be displayed at the Venue including on LED screens and no sponsor logos or branding shall appear on-screen at any time, including during pre-rolls, holding slides, the Artist's performance, or any post-show visuals."},
    {"id": 11, "text": "There shall be no branding on the three audience-facing sides of the venue excluding the stage by any third-party sponsor, festival, or college."},
]

# ── Run Consumer ──
print("=" * 72)
print("  CONSUMER CONTRACT ANALYSIS")
print("=" * 72)
result_c = analyze_clauses(consumer_clauses, "consumer", "Consumer SaaS Agreement")

print("\n  PER-CLAUSE RESULTS:")
for c in result_c["clauses"]:
    icon = {"low": "LOW", "medium": "MED", "high": "HI!"}[c["severity"]]
    print(f"    [{icon:>3}] S_doc={c['s_doc']:.3f} | Clause {c['id']:>2}: {c['text'][:65]}...")

# ── Run Company ──
print("\n\n")
print("=" * 72)
print("  COMPANY SPONSORSHIP AGREEMENT ANALYSIS")
print("=" * 72)
result_b = analyze_clauses(company_clauses, "company", "Brand Sponsorship Agreement")

print("\n  PER-CLAUSE RESULTS:")
for c in result_b["clauses"]:
    icon = {"low": "LOW", "medium": "MED", "high": "HI!"}[c["severity"]]
    print(f"    [{icon:>3}] S_doc={c['s_doc']:.3f} | Clause {c['id']:>2}: {c['text'][:65]}...")

# Check clause 8 specifically
print("\n\n  === CLAUSE 8 DEEP DIVE ===")
c8 = result_b["clauses"][7]
print(f"  Severity: {c8['severity'].upper()}")
print(f"  S_doc:    {c8['s_doc']}")
print(f"  Scores:   {c8['scores']}")
for rp in result_b["riskyParts"]:
    if "shall be required to register" in rp["text"] or "3,000" in rp["text"]:
        print(f"  Reason:   {rp['reason']}")

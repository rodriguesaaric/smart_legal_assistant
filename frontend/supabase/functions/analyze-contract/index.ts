import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });

  try {
    const { contractText, contractSource, userType } = await req.json();
    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) throw new Error("LOVABLE_API_KEY is not configured");

    const systemPrompt = `You are a legal contract risk analysis AI. Analyze contracts and return a JSON response.

User context:
- Contract source: ${contractSource || "unknown"}
- User type: ${userType || "consumer"}

Thresholds differ by user type:
- Consumer: lower tolerance for ambiguity (>40 is risky), malicious intent (>20 is risky)
- Company: higher tolerance for ambiguity (>60 is risky), malicious intent (>40 is risky)

You MUST respond with ONLY valid JSON in this exact format:
{
  "scores": {
    "ambiguity": <number 0-100>,
    "maliciousIntent": <number 0-100>,
    "clarity": <number 0-100>,
    "compliance": <number 0-100>
  },
  "overallRisk": "<low|medium|high|critical>",
  "summary": "<brief 2-3 sentence summary of findings>",
  "riskyParts": [
    {
      "text": "<exact quote from the contract>",
      "reason": "<why this is risky>",
      "severity": "<low|medium|high>",
      "category": "<ambiguity|malicious_intent|clarity|compliance>"
    }
  ]
}

Score meanings:
- ambiguity: 0=perfectly clear, 100=extremely ambiguous
- maliciousIntent: 0=no malicious intent, 100=clearly malicious
- clarity: 0=very unclear, 100=perfectly clear
- compliance: 0=non-compliant, 100=fully compliant

Identify 3-10 risky parts with exact quotes from the contract text. Be thorough.`;

    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-3-flash-preview",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: `Analyze this contract:\n\n${contractText}` },
        ],
      }),
    });

    if (!response.ok) {
      const status = response.status;
      if (status === 429) return new Response(JSON.stringify({ error: "Rate limited, please try again later." }), { status: 429, headers: { ...corsHeaders, "Content-Type": "application/json" } });
      if (status === 402) return new Response(JSON.stringify({ error: "Payment required." }), { status: 402, headers: { ...corsHeaders, "Content-Type": "application/json" } });
      throw new Error(`AI gateway error: ${status}`);
    }

    const data = await response.json();
    const content = data.choices?.[0]?.message?.content || "";
    
    // Parse JSON from response (handle markdown code blocks)
    let parsed;
    try {
      const jsonStr = content.replace(/```json\n?/g, "").replace(/```\n?/g, "").trim();
      parsed = JSON.parse(jsonStr);
    } catch {
      throw new Error("Failed to parse AI response as JSON");
    }

    return new Response(JSON.stringify(parsed), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (e) {
    console.error("analyze-contract error:", e);
    return new Response(JSON.stringify({ error: e instanceof Error ? e.message : "Unknown error" }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});

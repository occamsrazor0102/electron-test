import { useNavigate } from "react-router-dom";

const FAQ_ITEMS = [
  {
    q: "What does BEET do?",
    a: "BEET analyzes the writing patterns in text to determine whether it was written by a human or an AI. It looks at vocabulary patterns, sentence structure, statistical properties, and other signals that differ between human and AI writing.",
  },
  {
    q: "What do the colors mean?",
    a: null, // rendered specially
    special: "colors",
  },
  {
    q: "How accurate is this?",
    a: "BEET is a detection aid, not a perfect judge. Accuracy varies depending on text length, writing style, and which AI model may have been used. Short texts (under 150 words) are harder to analyze reliably. We recommend treating results as one signal among several, not as a final verdict.",
  },
  {
    q: "Can AI text get past this?",
    a: "Yes. Sophisticated users can paraphrase AI output or use techniques to evade detection. BEET is designed to catch common patterns, not to be adversarially robust. Results should be used as a screening tool, not a final determination.",
  },
  {
    q: "What does 'Confidence' mean?",
    a: "Confidence reflects how certain the analysis is. A wide range means the system is less sure — the true value could be anywhere in that range. A narrow range means higher certainty. High confidence doesn't mean the result is correct; it means the analysis was consistent.",
  },
  {
    q: "Why was my text flagged when it's human-written?",
    a: "False positives happen. Some humans write in styles that share patterns with AI output — highly formal writing, instructional content, or very structured text may look AI-like to the detector. A YELLOW or AMBER result doesn't mean the text is AI-generated; it means it's worth a second look.",
  },
  {
    q: "What should I do with a YELLOW result?",
    a: "Use your judgment. YELLOW means minor patterns were detected that can appear in both human and AI writing. If you know the author and the context, that information should carry weight. Consider requesting clarification from the author if needed.",
  },
  {
    q: "Is my text stored anywhere?",
    a: "When using the web interface, your text is sent to the BEET server for analysis. If you're running BEET locally, it stays on your computer. History is stored only in your browser's local storage and is never uploaded. Check with your organization's privacy policy if using a shared server.",
  },
  {
    q: "How do I check many files at once?",
    a: null,
    special: "batch",
  },
];

const COLOR_GUIDE = [
  { color: "bg-green-500", label: "GREEN", text: "Likely Human-Written", description: "No significant AI patterns detected." },
  { color: "bg-yellow-400", label: "YELLOW", text: "Some Patterns Detected", description: "Minor patterns that can appear in human writing too." },
  { color: "bg-orange-500", label: "AMBER", text: "Notable AI Patterns", description: "Patterns are notable — human review recommended." },
  { color: "bg-red-600", label: "RED", text: "Strong AI Indicators", description: "Strong indicators of AI generation." },
  { color: "bg-gray-400", label: "UNCERTAIN", text: "Unclear Result", description: "Analysis couldn't reach a clear conclusion." },
  { color: "bg-purple-500", label: "MIXED", text: "Mixed Authorship", description: "Parts of the text may be AI-generated." },
];

function FaqItem({ q, a, special }: { q: string; a: string | null; special?: string }) {
  const navigate = useNavigate();
  return (
    <details className="rounded-xl border border-gray-200 bg-white">
      <summary className="cursor-pointer px-5 py-4 font-semibold text-gray-900 hover:bg-gray-50 list-none flex items-center justify-between">
        {q}
        <span className="text-gray-400 text-sm" aria-hidden="true">▾</span>
      </summary>
      <div className="border-t border-gray-100 px-5 py-4 text-sm text-gray-600">
        {special === "colors" && (
          <ul className="space-y-2">
            {COLOR_GUIDE.map((c) => (
              <li key={c.label} className="flex items-start gap-3">
                <span
                  className={`mt-1 h-4 w-4 shrink-0 rounded-full ${c.color}`}
                  aria-hidden="true"
                />
                <span>
                  <strong>{c.text}</strong> — {c.description}
                </span>
              </li>
            ))}
          </ul>
        )}
        {special === "batch" && (
          <p>
            Use the{" "}
            <button
              onClick={() => navigate("/batch")}
              className="text-[#0D7377] underline hover:text-[#0b5e62] focus:outline-none"
            >
              Batch Analysis
            </button>{" "}
            page. You can drag-and-drop multiple .txt, .md, or .csv files and
            analyze them all at once. Results are shown in a table that you can
            export to CSV.
          </p>
        )}
        {a && <p>{a}</p>}
      </div>
    </details>
  );
}

export function HelpPage() {
  return (
    <div className="min-h-screen bg-[#FAFAF8]">
      <main className="mx-auto max-w-2xl px-4 py-10 space-y-4">
        <h1 className="text-2xl font-bold text-gray-900 mb-6">Help & FAQ</h1>
        {FAQ_ITEMS.map((item) => (
          <FaqItem key={item.q} {...item} />
        ))}
      </main>
    </div>
  );
}

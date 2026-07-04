import { useStore, type View } from "../store";
import { Badge } from "./ui";

const TABS: { v: View; label: string }[] = [
  { v: "map", label: "Map" },
  { v: "tensor", label: "Tensor" },
  { v: "td3", label: "3D" },
  { v: "journey", label: "Journey" },
];

export function TopBar() {
  const { model, setModel, run, view, setView, journey, loading } = useStore();
  const r = journey?.result;
  return (
    <header className="flex items-center gap-4 h-[60px] px-6 border-b border-bd bg-s1/60 backdrop-blur-md sticky top-0 z-10">
      <span className="flex items-center gap-2 font-extrabold tracking-tight">
        <span className="w-5 h-5 rounded-md bg-gradient-to-br from-ac to-acd" />
        orka
      </span>

      <div className="ml-5 flex gap-0.5 bg-[#100E18] border border-bd rounded-[9px] p-[3px]">
        {TABS.map((t) => (
          <button
            key={t.v}
            onClick={() => setView(t.v)}
            className={
              "text-[12px] px-3.5 py-1.5 rounded-md transition " +
              (view === t.v ? "bg-s3 text-tx border border-bd2" : "text-mut hover:text-tx")
            }
          >
            {t.label}
          </button>
        ))}
      </div>

      <input
        value={model}
        onChange={(e) => setModel(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && run()}
        placeholder="owner/model — press Enter"
        spellCheck={false}
        className="ml-auto w-[300px] bg-[#16131F] border border-bd2 rounded-[10px] px-3 py-2 text-[13px] text-tx placeholder:text-dim outline-none focus:border-ac/50 mono"
      />

      {loading && <span className="text-[12px] text-mut animate-pulse">analyzing…</span>}
      {r && !loading && (
        <>
          <span className="text-[12px] text-mut">
            compression <b className="mono text-ac ml-1">{r.ratio}×</b>
          </span>
          <span className="text-[12px] text-mut">
            ppl <b className="mono text-tx ml-1">{r.ppl_ratio ?? "—"}</b>
          </span>
          <Badge variant={r.source === "measured" ? "ok" : "mut"}>{r.source}</Badge>
        </>
      )}
    </header>
  );
}

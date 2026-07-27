import { useStore } from "../store";
import { Switch } from "./ui";

export function Rail() {
  const {
    journey, bpw, setBpw, keepHead, lattice, toggleKeepHead, toggleLattice,
    run, loading, runLive, stopLive, packing,
  } = useStore();
  const tricks = journey?.tricks ?? [];
  const toggles = tricks.filter((t) => t.kind === "toggle");
  const scalars = tricks.filter((t) => t.kind === "scalar" && t.id !== "bpw");

  // Only keep_head + lattice change the static estimate; the rest are GPU-run-only -> read-only.
  const interactive: Record<string, { on: boolean; fn: () => void }> = {
    keep_head_fp16: { on: keepHead, fn: toggleKeepHead },
    lattice: { on: lattice, fn: toggleLattice },
  };

  return (
    <aside className="bg-s1 border-r border-bd p-4 flex flex-col gap-6 overflow-auto">
      <div>
        <div className="ov mb-3">Trick Lab</div>
        {!journey && <p className="text-[11px] text-dim leading-relaxed">Analyze a model to populate.</p>}

        <div className="mb-4">
          <div className="flex justify-between text-[12.5px] text-mut mb-2">
            <span>Bits / weight</span>
            <b className="mono text-ac">{bpw.toFixed(2)}</b>
          </div>
          <input
            type="range" min={2.5} max={4} step={0.25} value={bpw}
            onChange={(e) => setBpw(parseFloat(e.target.value))}
            className="w-full accent-ac"
          />
        </div>

        {scalars.map((t) => (
          <div key={t.id} className="flex justify-between text-[12.5px] text-mut mb-2">
            <span>{t.label}</span>
            <b className="mono text-mut">{String(t.default)}</b>
          </div>
        ))}
      </div>

      <div className="-mt-2">
        {toggles.map((t) => {
          // `applies` is the backend's per-architecture verdict: false means toggling it
          // would change nothing for THIS model, so it is not offered as a control.
          const it = t.applies ? interactive[t.id] : undefined;
          const label = !t.applies ? "n/a here" : it ? null : "GPU-run";
          return (
            <div key={t.id} className="flex justify-between items-center py-2.5 text-[12.5px] border-t border-bd first:border-t-0">
              <span
                title={t.applies ? t.why : `${t.label}: ${t.why}`}
                className={(it ? (it.on ? "text-mut" : "text-dim") : "text-dim") + " flex items-center gap-1.5"}
              >
                {t.label}
                {label && <span className="text-[9px] text-dim border border-bd rounded px-1">{label}</span>}
                {t.warn && <span className="text-[10px] text-warn" title={t.warn}>⚠</span>}
              </span>
              <Switch on={it ? it.on : Boolean(t.default)} onClick={it?.fn} />
            </div>
          );
        })}
      </div>

      <div className="mt-auto flex flex-col gap-2">
        <button
          onClick={() => run()}
          disabled={loading || packing}
          className="font-semibold text-[13px] text-[#0B0A11] bg-ac rounded-[9px] py-3 disabled:opacity-50"
        >
          {loading ? "analyzing…" : "Re-analyze"}
        </button>
        <button
          onClick={() => (packing ? stopLive() : runLive())}
          disabled={loading}
          title="Pack + evaluate on the GPU and replace the estimate with measured numbers"
          className={
            "font-semibold text-[13px] rounded-[9px] py-3 border disabled:opacity-50 " +
            (packing
              ? "border-warn/40 text-warn hover:bg-warn/10"
              : "border-ok/40 text-ok hover:bg-ok/10")
          }
        >
          {packing ? "stop watching…" : "Run on GPU (measured)"}
        </button>
      </div>
    </aside>
  );
}

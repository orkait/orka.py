import { useStore } from "../store";
import { Switch } from "./ui";

export function Rail() {
  const { journey, bpw, setBpw, run, loading } = useStore();
  const tricks = journey?.tricks ?? [];
  const scalars = tricks.filter((t) => t.kind === "scalar");
  const toggles = tricks.filter((t) => t.kind === "toggle");

  return (
    <aside className="bg-s1 border-r border-bd p-4 flex flex-col gap-6 overflow-auto">
      <div>
        <div className="ov mb-3">Trick Lab</div>
        {!journey && <p className="text-[11px] text-dim leading-relaxed">Analyze a model to populate.</p>}

        {scalars.map((t) => (
          <div key={t.id} className="mb-4">
            <div className="flex justify-between text-[12.5px] text-mut mb-2">
              <span>{t.label}</span>
              <b className="mono text-ac">{t.id === "bpw" ? bpw.toFixed(2) : String(t.default)}</b>
            </div>
            {t.id === "bpw" ? (
              <input
                type="range" min={2.5} max={4} step={0.25} value={bpw}
                onChange={(e) => setBpw(parseFloat(e.target.value))}
                onMouseUp={() => run()}
                className="w-full accent-ac"
              />
            ) : (
              <div className="h-[5px] bg-[#1C1928] rounded-full">
                <span className="block h-[5px] rounded-full bg-ac" style={{ width: "40%" }} />
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="-mt-2">
        {toggles.map((t) => (
          <div key={t.id} className="flex justify-between items-center py-2.5 text-[12.5px] border-t border-bd first:border-t-0">
            <span className={(t.default ? "text-mut" : "text-dim") + " flex items-center gap-1.5"}>
              {t.label}
              {t.warn && <span className="text-[10px] text-warn">⚠</span>}
            </span>
            <Switch on={Boolean(t.default)} />
          </div>
        ))}
      </div>

      <button
        onClick={() => run()}
        disabled={loading}
        className="mt-auto font-semibold text-[13px] text-[#0B0A11] bg-ac rounded-[9px] py-3 disabled:opacity-50"
      >
        Run for real — GPU
      </button>
    </aside>
  );
}

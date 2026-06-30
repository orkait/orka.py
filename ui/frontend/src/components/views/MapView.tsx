import { useStore } from "../../store";
import { Badge, Card } from "../ui";
import type { ModuleEntry } from "../../types";

const short = (n: string) => n.replace(/\.weight$/, "").split(".").slice(-2).join(".");
const treatColor = (t: ModuleEntry["treatment"]) =>
  t === "keep_fp16"
    ? "bg-crit/15 text-crit border border-crit/30"
    : t === "skip_error_comp"
    ? "bg-warn/15 text-warn border border-warn/30"
    : "bg-ac2 text-white border border-transparent";

export function MapView() {
  const { journey, selectTensor } = useStore();
  if (!journey) return null;
  const a = journey.architecture;
  const head = a.layers.find((l) => l.index === -1)?.modules ?? [];
  const layers = a.layers.filter((l) => l.index >= 0);

  return (
    <div className="p-6 flex flex-col gap-6 overflow-auto h-full">
      <div className="flex items-center gap-3">
        <h2 className="text-[16px] font-semibold">Architecture map</h2>
        <Badge variant="mut">{a.arch_class}</Badge>
        {a.flags.tied_head && <Badge variant="crit">tied head</Badge>}
        {a.flags.has_moe && <Badge variant="ac">MoE</Badge>}
        {a.flags.has_ssm && <Badge variant="warn">SSM</Badge>}
        <span className="ml-auto mono text-[12px] text-dim">
          {layers.length} layers · {(journey.model.params_total / 1e6).toFixed(0)}M params
        </span>
      </div>

      <div className="flex items-stretch gap-2 overflow-x-auto pb-2">
        {head.filter((m) => /embed/i.test(m.name)).slice(0, 1).map((m) => (
          <IoBlock key={m.name} label="embed" />
        ))}
        {layers.slice(0, 16).map((l) => (
          <div key={l.index} className="min-w-[112px] bg-s2 border border-bd rounded-xl p-2.5 flex flex-col gap-1.5">
            <span className="text-[10px] text-dim font-semibold mono">L{l.index}</span>
            {l.modules.slice(0, 6).map((m) => (
              <button
                key={m.name}
                onClick={() => selectTensor(m.name)}
                title={`${m.name}  ${m.shape.join("×")}  ${m.treatment}`}
                className={"text-[10.5px] rounded-md px-2 py-1 text-left truncate transition hover:brightness-125 " + treatColor(m.treatment)}
              >
                {short(m.name)}
              </button>
            ))}
          </div>
        ))}
        {layers.length > 16 && <span className="self-center text-dim px-1">…</span>}
        {head.some((m) => /lm_head|embed_out|output/i.test(m.name)) && <IoBlock label="head" />}
      </div>

      <Card title="Parameter breakdown">
        <div className="flex flex-col gap-2.5 mt-1">
          {a.param_breakdown.map((f) => (
            <div key={f.family}>
              <div className="flex justify-between text-[12px] text-mut">
                <span>{f.family}</span>
                <span className="mono">{f.pct}%</span>
              </div>
              <div className="h-1.5 rounded bg-bd mt-1">
                <span className="block h-1.5 rounded bg-gradient-to-r from-acd to-ac" style={{ width: `${f.pct}%` }} />
              </div>
            </div>
          ))}
        </div>
        <p className="text-[12px] text-mut mt-3 leading-relaxed">
          ↳ click any tensor to open its deep-detail. Pink = kept fp16, amber = error-comp skipped (SSM/head), violet = quantized.
        </p>
      </Card>
    </div>
  );
}

function IoBlock({ label }: { label: string }) {
  return (
    <div className="min-w-[78px] flex flex-col justify-center items-center gap-1.5 bg-crit/[0.07] border border-crit/30 rounded-xl p-3 text-crit text-[12px] font-semibold">
      {label}
      <small className="text-dim font-normal text-[10px]">fp16</small>
    </div>
  );
}

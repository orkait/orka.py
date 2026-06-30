import { useStore } from "../../store";
import { Badge, Card } from "../ui";
import { ArchFlow } from "../ArchFlow";

export function MapView() {
  const { journey } = useStore();
  if (!journey) return null;
  const a = journey.architecture;
  const layers = a.layers.filter((l) => l.index >= 0);

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center gap-3 px-6 pt-5 pb-3">
        <h2 className="text-[16px] font-semibold">Architecture map</h2>
        <Badge variant="mut">{a.arch_class}</Badge>
        {a.flags.tied_head && <Badge variant="crit">tied head</Badge>}
        {a.flags.has_moe && <Badge variant="ac">MoE</Badge>}
        {a.flags.has_ssm && <Badge variant="warn">SSM</Badge>}
        <span className="ml-auto mono text-[12px] text-dim">
          {layers.length} layers · {(journey.model.params_total / 1e6).toFixed(0)}M params · scroll to zoom, drag to pan
        </span>
      </div>

      <div className="flex-1 min-h-0 mx-6 rounded-xl border border-bd overflow-hidden">
        <ArchFlow />
      </div>

      <div className="px-6 py-4 shrink-0">
        <Card title="Parameter breakdown">
          <div className="grid grid-cols-2 gap-x-8 gap-y-2 mt-1">
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
            ↳ identical layers collapse into one <span className="text-ac">×N</span> group. Click any tensor chip to deep-probe it · pink = kept fp16 · amber = error-comp skipped · violet = quantized.
          </p>
        </Card>
      </div>
    </div>
  );
}

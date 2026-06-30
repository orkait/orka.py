import { useStore } from "../../store";
import { Badge, Card, Stat } from "../ui";

export function TensorView() {
  const { journey, selectedTensor } = useStore();
  if (!journey) return null;
  const r = journey.result;
  const mod = journey.architecture.layers.flatMap((l) => l.modules).find((m) => m.name === selectedTensor);

  return (
    <div className="p-8 overflow-auto h-full">
      <div className="flex items-baseline gap-3 mb-2">
        <span className="mono text-[18px] font-semibold tracking-tight">{mod ? mod.name : "Select a tensor from the Map"}</span>
        {mod && <span className="text-[12.5px] text-dim mono">{mod.shape.join(" × ")} · {mod.family}</span>}
      </div>
      {mod && (
        <div className="flex gap-2 mb-6">
          <Badge variant="ac">{mod.treatment}</Badge>
        </div>
      )}

      <div className="flex gap-8 py-6 my-6 border-y border-bd">
        <Stat value={String(r.ratio)} unit="×" label="COMPRESSION" color="var(--color-ac)" />
        <Stat value={r.ppl_ratio ? String(r.ppl_ratio) : "—"} unit="×" label="PPL RATIO" color="var(--color-ok)" />
        <Stat value={r.fp16_mb.toFixed(0)} unit="MB" label="FP16 SIZE" color="var(--color-mut)" />
        <Stat value={r.orka_mb.toFixed(0)} unit="MB" label="ORKA SIZE" />
      </div>

      <div className="grid grid-cols-2 gap-6">
        {[
          ["Weight matrix · heatmap", "real sampled bytes (range-fetch)"],
          ["Value distribution + codebook ticks", "where the 256 codewords land"],
          ["Rate–distortion curve", "why this bpw was chosen"],
          ["Quant error · residual map", "where error concentrates"],
        ].map(([t, sub]) => (
          <Card key={t} title={t} className="min-h-[180px]">
            <div className="flex-1 flex items-center justify-center text-center">
              <div>
                <div className="text-[13px] text-mut">{sub}</div>
                <Badge variant="warn" className="mt-3">needs deep-probe · sampling — next</Badge>
              </div>
            </div>
          </Card>
        ))}
      </div>

      <Card title="Notes" className="mt-6">
        <ul className="text-[12.5px] text-mut leading-relaxed list-disc pl-4">
          {r.notes.map((n, i) => <li key={i}>{n}</li>)}
        </ul>
      </Card>
    </div>
  );
}

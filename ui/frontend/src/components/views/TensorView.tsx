import { useStore } from "../../store";
import { Badge, Card, Stat } from "../ui";
import type { TensorProbe } from "../../types";

const WRAMP = ["#5B54FF", "#473F9E", "#2C2748", "#1F1B30", "#15131F", "#2C2748", "#473F9E", "#7A6CE8", "#A99BFF"];
const wColor = (v: number) => WRAMP[Math.max(0, Math.min(8, Math.round((v + 1) / 2 * 8)))];
const eColor = (v: number) =>
  v > 0.66 ? "#F472A6" : v > 0.4 ? "#9C3A63" : v > 0.18 ? "#4A2236" : v > 0.06 ? "#1E1620" : "#0D0B12";

function Heat({ block, color, cols }: { block: number[][]; color: (v: number) => string; cols: number }) {
  return (
    <div className="grid gap-px flex-1 min-h-0" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
      {block.flat().map((v, i) => (
        <span key={i} className="rounded-[1px]" style={{ aspectRatio: "1", background: color(v) }} />
      ))}
    </div>
  );
}

export function TensorView() {
  const { journey, selectedTensor, probe, probeLoading, probeError } = useStore();
  if (!journey) return null;
  const mod = journey.architecture.layers.flatMap((l) => l.modules).find((m) => m.name === selectedTensor);

  return (
    <div className="p-8 overflow-auto h-full">
      <div className="flex items-baseline gap-3 mb-2">
        <span className="mono text-[18px] font-semibold tracking-tight">{selectedTensor ?? "Select a tensor from the Map"}</span>
        {probe && <span className="text-[12.5px] text-dim mono">{probe.shape.join(" × ")} · {probe.dtype} · sampled {probe.sampled_elems.toLocaleString()}</span>}
      </div>
      {mod && <div className="flex gap-2 mb-6"><Badge variant="ac">{mod.treatment}</Badge><Badge variant="mut">{mod.family}</Badge></div>}

      {!selectedTensor && <div className="text-dim mt-10">Open the Map and click any tensor node to deep-probe it.</div>}
      {probeLoading && <div className="text-mut animate-pulse mt-10">range-fetching weights + running VQ on the sample…</div>}
      {probeError && <div className="rounded-xl border border-crit/30 bg-crit/[0.07] text-crit px-4 py-3 text-[13px] mt-4">{probeError}</div>}

      {probe && <Body p={probe} ratio={journey.result.ratio} />}
    </div>
  );
}

function Body({ p, ratio }: { p: TensorProbe; ratio: number }) {
  const span = p.dist_range[1] - p.dist_range[0] || 1;
  const pos = (v: number) => `${Math.max(0, Math.min(100, ((v - p.dist_range[0]) / span) * 100))}%`;
  const maxSqnr = Math.max(...p.rd.map((d) => d.sqnr), 1) * 1.1;
  const rdPts = p.rd.map((d) => `${((d.bpw - 1) / 3) * 116 + 8},${58 - (d.sqnr / maxSqnr) * 52}`).join(" ");

  return (
    <>
      <div className="flex gap-10 py-6 my-6 border-y border-bd">
        <Stat value={String(ratio)} unit="×" label="MODEL COMPRESSION" color="var(--color-ac)" />
        <Stat value={String(p.sqnr_3bpw)} unit="dB" label="SQNR @ 3 BPW" color="var(--color-ok)" />
        <Stat value={String(p.error_pct)} unit="%" label="QUANT ERROR" />
        <Stat value={String(p.outlier_pct)} unit="%" label="OUTLIERS (>3σ)" color="var(--color-warn)" />
      </div>

      <div className="grid grid-cols-2 gap-6 items-start">
        <div className="flex flex-col gap-6">
          <Card title="Weight matrix · sampled window"
            right={<span className="flex items-center gap-1.5 text-[10px] text-dim mono">−<span className="w-20 h-2 rounded" style={{ background: "linear-gradient(90deg,#5B54FF,#15131F,#A99BFF)" }} />+</span>}>
            <Heat block={p.weights_block} color={wColor} cols={p.weights_block[0]?.length ?? 40} />
            <p className="text-[12px] text-ok mt-3 leading-relaxed">↳ real bytes, range-fetched. std <span className="mono">{p.std}</span>, range <span className="mono">[{p.vmin}, {p.vmax}]</span>.</p>
          </Card>
          <Card title="Quant error · residual @ 3 bpw"
            right={<span className="w-20 h-2 rounded" style={{ background: "linear-gradient(90deg,#0D0B12,#9C3A63,#F472A6)" }} />}>
            <Heat block={p.error_block} color={eColor} cols={p.error_block[0]?.length ?? 40} />
            <p className="text-[12px] text-ok mt-3 leading-relaxed">↳ where the 3 bpw reconstruction misses — brighter = larger residual.</p>
          </Card>
        </div>

        <div className="flex flex-col gap-6">
          <Card title="Value distribution" right={<span className="mono text-[11px] text-dim">μ {p.mean} · σ {p.std}</span>}>
            <div className="relative">
              <div className="absolute top-0 h-[110px] bg-ac/[0.06] border-x border-dashed border-ac/30"
                style={{ left: pos(p.mean - p.std), width: `calc(${pos(p.mean + p.std)} - ${pos(p.mean - p.std)})` }} />
              <div className="flex items-end gap-px h-[110px]">
                {p.distribution.map((h, i) => <span key={i} className="flex-1 rounded-t-[2px] bg-ac2" style={{ height: `${Math.max(2, h * 100)}%` }} />)}
              </div>
              <div className="relative h-3 mt-1">
                {p.codebook_values.map((v, i) => <span key={i} className="absolute w-px h-2 bg-ok" style={{ left: pos(v) }} />)}
              </div>
            </div>
            <p className="text-[12px] text-mut mt-2 leading-relaxed"><span className="text-ok">green ticks</span> = where codewords land; <span className="text-ac">band</span> = ±σ.</p>
          </Card>

          <Card title="Rate–distortion · RVQ on sample" right={<Badge variant="ok">{p.sqnr_3bpw} dB @ 3</Badge>}>
            <svg viewBox="0 0 130 64" preserveAspectRatio="none" className="h-[110px]">
              <g stroke="#221F2E" strokeWidth=".6"><line x1="8" y1="14" x2="128" y2="14" /><line x1="8" y1="32" x2="128" y2="32" /><line x1="8" y1="50" x2="128" y2="50" /></g>
              <polyline points={rdPts} fill="none" stroke="var(--color-ac)" strokeWidth="2.5" strokeLinecap="round" />
              {p.rd.map((d) => <circle key={d.bpw} cx={((d.bpw - 1) / 3) * 116 + 8} cy={58 - (d.sqnr / maxSqnr) * 52} r={d.bpw === 3 ? 4 : 2.5} fill={d.bpw === 3 ? "var(--color-ok)" : "#3A3550"} stroke="#08070D" strokeWidth="1" />)}
            </svg>
            <div className="flex justify-between text-[10px] text-dim mono mt-1 px-2"><span>1</span><span>2</span><span>3 bpw</span><span>4</span></div>
            <p className="text-[12px] text-ok mt-2 leading-relaxed">↳ SQNR climbs ~{((p.rd[3]?.sqnr - p.rd[0]?.sqnr) / 3).toFixed(1)} dB/bpw — diminishing returns set the knee.</p>
          </Card>

          <Card title="Codebook utilization" right={<Badge variant={p.entropy_bits > p.entropy_max * 0.92 ? "warn" : "ok"}>H {p.entropy_bits}/{p.entropy_max}</Badge>}>
            <div className="flex items-end gap-px h-[70px]">
              {p.utilization.map((h, i) => <span key={i} className="flex-1 rounded-t-[2px]" style={{ height: `${Math.max(2, h * 100)}%`, background: i < 13 ? "var(--color-ac)" : i < 20 ? "var(--color-acd)" : "#3A3550" }} />)}
            </div>
            <p className="text-[12px] text-mut mt-2 leading-relaxed">↳ entropy {p.entropy_bits}/{p.entropy_max} bits → {p.entropy_bits > p.entropy_max * 0.92 ? "near-max, ANS/zip won't shrink indices" : "headroom for entropy coding"}.</p>
          </Card>
        </div>
      </div>
    </>
  );
}

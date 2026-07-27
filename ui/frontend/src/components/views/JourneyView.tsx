import { useStore } from "../../store";
import { Badge, Card, Stat } from "../ui";

// The live runner reports coarse phases; map them onto the pipeline steps they cover so the
// stepper reflects a real run instead of a hardcoded position.
const LIVE_TO_STEPS: Record<string, string[]> = {
  download: ["load"],
  pack: ["transform", "allocate", "codebook", "quantize", "strategies", "pack"],
  eval: [],
};

export function JourneyView() {
  const { journey, liveLog, liveError, packing } = useStore();
  if (!journey) return null;
  const steps = journey.pipeline;
  const r = journey.result;

  const seen = liveLog.map((e) => e.stage);
  const activeStage = seen.length ? seen[seen.length - 1] : null;
  const activeSteps = new Set(activeStage ? LIVE_TO_STEPS[activeStage] ?? [] : []);
  const doneSteps = new Set(
    seen.slice(0, -1).flatMap((s) => LIVE_TO_STEPS[s] ?? []),
  );
  // Everything the pack phase covers is finished once eval starts.
  if (activeStage === "eval") for (const s of LIVE_TO_STEPS.pack) doneSteps.add(s);

  const state = (id: string) =>
    activeSteps.has(id) ? "active" : doneSteps.has(id) ? "done" : "idle";

  return (
    <div className="p-6 flex flex-col gap-6 overflow-auto h-full">
      <div className="flex items-center gap-3">
        <h2 className="text-[16px] font-semibold">Compression journey</h2>
        <Badge variant="ac">rvq-12-12 + em-aq + hessian</Badge>
        <Badge variant={r.source === "measured" ? "ok" : "mut"}>{r.source}</Badge>
        {packing && <span className="text-[12px] text-ok animate-pulse">running on GPU…</span>}
      </div>

      {liveError && (
        <div className="rounded-xl border border-crit/30 bg-crit/[0.07] text-crit px-4 py-3 text-[13px]">
          {liveError}
        </div>
      )}

      <div className="flex items-start">
        {steps.map((s, i) => {
          const st = state(s.id);
          return (
            <div key={s.id} className="flex-1 text-center relative">
              {i < steps.length - 1 && (
                <span className={"absolute top-[17px] left-1/2 w-full h-0.5 " + (st === "done" ? "bg-ok" : "bg-bd")} />
              )}
              <div
                className={
                  "w-[34px] h-[34px] rounded-full mx-auto mb-1.5 flex items-center justify-center mono text-[13px] relative z-10 " +
                  (st === "active"
                    ? "bg-gradient-to-br from-ac to-acd text-[#0B0A11]"
                    : st === "done"
                    ? "bg-ok/15 border border-ok text-ok"
                    : "bg-s2 border border-bd2 text-mut")
                }
              >
                {i + 1}
              </div>
              <div className={"text-[11px] " + (st === "active" ? "text-tx font-semibold" : "text-mut")}>{s.title}</div>
            </div>
          );
        })}
      </div>

      {!seen.length && (
        <p className="text-[12px] text-dim leading-relaxed -mt-2">
          Stage state is idle until a GPU run reports progress. Press{" "}
          <span className="text-ok">Run on GPU</span> to pack + evaluate this model and
          replace the estimate with measured numbers.
        </p>
      )}

      <div className="grid grid-cols-3 gap-4">
        <Card title={r.source === "measured" ? "Measured result" : "Estimated result"}>
          <div className="flex gap-6">
            <Stat value={String(r.ratio)} unit="×" label="COMPRESSION" color="var(--color-ac)" />
            <Stat value={r.ppl_ratio == null ? "—" : String(r.ppl_ratio)} unit="×" label="PPL RATIO"
              color={r.source === "measured" ? "var(--color-ok)" : undefined} />
          </div>
          <p className="text-[12px] text-mut mt-3 leading-relaxed mono">
            {r.fp16_mb} MB → {r.orka_mb} MB at {r.bpw} bpw
          </p>
          {r.trusted === false && r.trust_reason && (
            <p className="text-[12px] text-warn mt-2 leading-relaxed">↳ {r.trust_reason}</p>
          )}
        </Card>

        <Card title="Live progress">
          {seen.length ? (
            <div className="flex flex-col gap-1.5 text-[12px]">
              {liveLog.map((e, i) => (
                <div key={i} className="flex justify-between text-mut">
                  <span className="mono text-ac">{e.stage}</span>
                  <span className="text-dim truncate ml-3">{e.msg ?? ""}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-[12px] text-dim leading-relaxed">No GPU run in this session.</p>
          )}
        </Card>

        <Card title="Why these numbers">
          <div className="flex flex-col gap-1.5 text-[12px] text-mut">
            {r.notes.length
              ? r.notes.map((n, i) => <span key={i}>↳ {n}</span>)
              : <span className="text-dim">no notes reported</span>}
          </div>
        </Card>
      </div>

      <Card title="Stages">
        <div className="grid grid-cols-2 gap-x-8 gap-y-1.5 text-[12px]">
          {steps.map((s) => (
            <div key={s.id} className="flex justify-between text-mut">
              <span title={s.summary}>{s.title}</span>
              <span className="mono text-dim">{state(s.id)}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

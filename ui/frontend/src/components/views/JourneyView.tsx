import { useStore } from "../../store";
import { Badge, Card } from "../ui";

export function JourneyView() {
  const { journey } = useStore();
  if (!journey) return null;
  const steps = journey.pipeline;

  return (
    <div className="p-6 flex flex-col gap-6 overflow-auto h-full">
      <div className="flex items-center gap-3">
        <h2 className="text-[16px] font-semibold">Compression journey</h2>
        <Badge variant="ac">rvq-12-12 + em-aq + hessian</Badge>
      </div>

      <div className="flex items-start">
        {steps.map((s, i) => {
          const done = i < 3, cur = i === 3;
          return (
            <div key={s.id} className="flex-1 text-center relative">
              {i < steps.length - 1 && (
                <span className={"absolute top-[17px] left-1/2 w-full h-0.5 " + (done ? "bg-ok" : "bg-bd")} />
              )}
              <div
                className={
                  "w-[34px] h-[34px] rounded-full mx-auto mb-1.5 flex items-center justify-center mono text-[13px] relative z-10 " +
                  (cur
                    ? "bg-gradient-to-br from-ac to-acd text-[#0B0A11]"
                    : done
                    ? "bg-ok/15 border border-ok text-ok"
                    : "bg-s2 border border-bd2 text-mut")
                }
              >
                {i + 1}
              </div>
              <div className={"text-[11px] " + (cur ? "text-tx font-semibold" : "text-mut")}>{s.title}</div>
            </div>
          );
        })}
      </div>

      <div className="grid grid-cols-3 gap-4">
        <Card title="Residual energy / stage">
          <div className="flex items-end gap-1 h-[90px]">
            <span className="flex-1 rounded-t bg-ac" style={{ height: "100%" }} />
            <span className="flex-1 rounded-t bg-acd" style={{ height: "34%" }} />
            <span className="flex-1 rounded-t bg-[#473F9E]" style={{ height: "11%" }} />
          </div>
          <p className="text-[12px] text-ok mt-3 leading-relaxed">↳ each RVQ stage soaks up leftover error — stage 3 residual ~3% of stage 1.</p>
        </Card>
        <Card title="Active stage">
          <div className="text-[13px] font-semibold text-ac">{journey.pipeline[3]?.title}</div>
          <p className="text-[12.5px] text-mut mt-2 leading-relaxed">{journey.pipeline[3]?.summary}</p>
          <p className="text-[12px] text-ok mt-3 leading-relaxed">↳ Hessian weighting pulls centroids toward output-sensitive directions — the big free quality lever.</p>
        </Card>
        <Card title="Stages">
          <div className="flex flex-col gap-1.5 text-[12px]">
            {steps.map((s) => (
              <div key={s.id} className="flex justify-between text-mut">
                <span>{s.title}</span>
                <span className="mono text-dim">{s.id}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

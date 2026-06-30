import { Badge } from "../ui";

export function ThreeDView() {
  return (
    <div className="p-6 h-full">
      <div className="h-full rounded-2xl border border-bd2 flex flex-col items-center justify-center gap-4"
        style={{ background: "radial-gradient(80% 80% at 50% 45%, rgba(139,124,246,.10), transparent 70%), #0A0810" }}>
        <span className="ov text-ac">3D codebook · vectors → centroids</span>
        <Badge variant="warn">needs weight sampling (deep-probe) + R3F scene — next</Badge>
        <p className="text-[12.5px] text-mut max-w-md text-center leading-relaxed">
          The 3D scene plots sampled weight vectors clustering to learned codebook centroids
          (and the E8 lattice). It rides on the same per-tensor sampling the Tensor view needs.
        </p>
      </div>
    </div>
  );
}

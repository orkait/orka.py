import { useEffect, useRef } from "react";
import { useStore } from "../../store";
import { Badge } from "../ui";

type V3 = [number, number, number];

function rotate([x, y, z]: V3, yaw: number, pitch: number): V3 {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cx = Math.cos(pitch), sx = Math.sin(pitch);
  const x1 = cy * x + sy * z, z1 = -sy * x + cy * z;
  return [x1, cx * y - sx * z1, sx * y + cx * z1];
}

function Scene({ vectors, centroids }: { vectors: number[][]; centroids: number[][] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const rot = useRef({ yaw: 0.6, pitch: -0.35 });
  const drag = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d")!;

    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = cv.clientWidth, h = cv.clientHeight;
      cv.width = w * dpr; cv.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const cx = w / 2, cy = h / 2, scale = Math.min(w, h) * 0.34;
      const { yaw, pitch } = rot.current;
      const pts: { px: number; py: number; z: number; size: number; color: string; cen: boolean }[] = [];
      for (const v of vectors) { const [x, y, z] = rotate(v as V3, yaw, pitch); pts.push({ px: cx + x * scale, py: cy + y * scale, z, size: 2.2, color: "#8B7CF6", cen: false }); }
      for (const c of centroids) { const [x, y, z] = rotate(c as V3, yaw, pitch); pts.push({ px: cx + x * scale, py: cy + y * scale, z, size: 5, color: "#6EE7B7", cen: true }); }
      pts.sort((a, b) => a.z - b.z);
      for (const { px, py, z, size, color, cen } of pts) {
        const depth = Math.max(0, Math.min(1, (z + 1.4) / 2.8));
        ctx.globalAlpha = cen ? 0.95 : 0.22 + depth * 0.55;
        ctx.shadowBlur = cen ? 12 : 0;
        ctx.shadowColor = "#6EE7B7";
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(px, py, size * (0.6 + depth * 0.8), 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1; ctx.shadowBlur = 0;
    };

    draw();
    const spin = window.setInterval(() => { if (!drag.current) { rot.current.yaw += 0.004; draw(); } }, 40);
    const down = (e: PointerEvent) => { drag.current = { x: e.clientX, y: e.clientY }; };
    const move = (e: PointerEvent) => {
      if (!drag.current) return;
      rot.current.yaw += (e.clientX - drag.current.x) * 0.01;
      rot.current.pitch = Math.max(-1.4, Math.min(1.4, rot.current.pitch + (e.clientY - drag.current.y) * 0.01));
      drag.current = { x: e.clientX, y: e.clientY };
      draw();
    };
    const up = () => { drag.current = null; };
    cv.addEventListener("pointerdown", down);
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    const ro = new ResizeObserver(draw);
    ro.observe(cv);
    return () => { clearInterval(spin); cv.removeEventListener("pointerdown", down); window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); ro.disconnect(); };
  }, [vectors, centroids]);

  return <canvas ref={ref} className="w-full h-full block cursor-grab active:cursor-grabbing" />;
}

export function ThreeDView() {
  const { probe, probeLoading, selectedTensor } = useStore();

  return (
    <div className="p-6 h-full">
      <div className="h-full rounded-2xl border border-bd2 relative overflow-hidden"
        style={{ background: "radial-gradient(80% 80% at 50% 45%, rgba(139,124,246,.10), transparent 70%), #0A0810" }}>
        <div className="absolute top-4 left-4 right-4 flex items-center gap-2 z-10 pointer-events-none">
          <span className="ov text-ac">3D codebook · vectors → centroids</span>
          {probe && <Badge variant="mut">PCA 8→3 · {probe.vectors3d.length} vecs · {probe.centroids3d.length} centroids</Badge>}
          <span className="ml-auto"><Badge variant="ac">drag to rotate</Badge></span>
        </div>

        {!selectedTensor && <Center>Pick a tensor in the Map to plot its codebook geometry.</Center>}
        {probeLoading && <Center>sampling + clustering…</Center>}
        {probe && <Scene vectors={probe.vectors3d} centroids={probe.centroids3d} />}

        {probe && (
          <div className="absolute bottom-4 left-4 flex gap-5 text-[11px] text-mut z-10">
            <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-ac2" />weight vectors (sampled)</span>
            <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-ok shadow-[0_0_8px_#6EE7B7]" />codebook centroids</span>
          </div>
        )}
      </div>
    </div>
  );
}

function Center({ children }: { children: React.ReactNode }) {
  return <div className="h-full flex items-center justify-center text-mut text-[13px] text-center px-8">{children}</div>;
}

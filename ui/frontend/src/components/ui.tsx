import type { ReactNode } from "react";
import { cn } from "../lib/cn";

type Variant = "mut" | "ac" | "ok" | "warn" | "crit";

export function Badge({ children, variant = "mut", className }: { children: ReactNode; variant?: Variant; className?: string }) {
  const v: Record<Variant, string> = {
    mut: "bg-s2 text-mut border-bd",
    ac: "bg-ac/10 text-ac border-ac/25",
    ok: "bg-ok/10 text-ok border-ok/25",
    warn: "bg-warn/10 text-warn border-warn/25",
    crit: "bg-crit/10 text-crit border-crit/25",
  };
  return (
    <span className={cn("inline-flex items-center gap-1.5 text-[11px] font-medium rounded-md px-2.5 py-1 border", v[variant], className)}>
      {children}
    </span>
  );
}

export function Card({ title, right, children, className }: { title?: string; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={cn("bg-s2 border border-bd rounded-xl p-5 flex flex-col min-h-0 transition duration-150 hover:-translate-y-0.5 hover:border-bd2", className)}>
      {title && (
        <div className="flex items-center justify-between mb-3">
          <span className="ov">{title}</span>
          {right}
        </div>
      )}
      {children}
    </section>
  );
}

export function Switch({ on, onClick }: { on: boolean; onClick?: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={!onClick}
      aria-pressed={on}
      className={cn(
        "w-8 h-[18px] rounded-full relative transition shrink-0",
        on ? "bg-ac" : "bg-[#26222F]",
        onClick ? "cursor-pointer hover:brightness-110" : "cursor-default opacity-70",
      )}
    >
      <i className={cn("absolute top-0.5 w-3.5 h-3.5 rounded-full bg-white transition-all", on ? "left-4" : "left-0.5")} />
    </button>
  );
}

export function Stat({ value, unit, label, color }: { value: string; unit?: string; label: string; color?: string }) {
  return (
    <div className="flex flex-col gap-2">
      <span className="mono text-[34px] font-bold tracking-tight leading-none" style={{ color }}>
        {value}
        {unit && <span className="text-[18px] text-mut ml-0.5">{unit}</span>}
      </span>
      <span className="text-[11px] text-dim tracking-wide">{label}</span>
    </div>
  );
}

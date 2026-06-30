import type { Journey } from "./types";

const API = (import.meta.env.VITE_API as string) ?? "http://127.0.0.1:8723";

export async function analyze(model: string, bpw = 3.0): Promise<Journey> {
  const r = await fetch(`${API}/analyze?model=${encodeURIComponent(model)}&bpw=${bpw}`);
  if (!r.ok) {
    const body = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(body.detail || `HTTP ${r.status}`);
  }
  return r.json();
}

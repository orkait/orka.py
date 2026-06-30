import type { Journey, TensorProbe } from "./types";

const API = (import.meta.env.VITE_API as string) ?? "http://127.0.0.1:8723";

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(`${API}${path}`);
  if (!r.ok) {
    const body = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(body.detail || `HTTP ${r.status}`);
  }
  return r.json();
}

export const analyze = (model: string, bpw = 3.0) =>
  getJSON<Journey>(`/analyze?model=${encodeURIComponent(model)}&bpw=${bpw}`);

export const probeTensor = (model: string, name: string) =>
  getJSON<TensorProbe>(`/tensor?model=${encodeURIComponent(model)}&name=${encodeURIComponent(name)}`);

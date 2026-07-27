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

// A journey is a pure function of these four knobs, and the bpw slider walks the same
// values repeatedly. Without a cache every step of a drag re-hit /analyze, which re-reads
// the safetensors header (2 range GETs per shard) on the server.
const journeyCache = new Map<string, Journey>();

export const journeyKey = (model: string, bpw: number, keepHead: boolean, lattice: boolean) =>
  `${model}|${bpw}|${keepHead}|${lattice}`;

export function cachedJourney(
  model: string, bpw: number, keepHead: boolean, lattice: boolean,
): Journey | undefined {
  return journeyCache.get(journeyKey(model, bpw, keepHead, lattice));
}

export async function analyze(
  model: string, bpw = 3.0, keepHead = true, lattice = false,
): Promise<Journey> {
  const key = journeyKey(model, bpw, keepHead, lattice);
  const hit = journeyCache.get(key);
  if (hit) return hit;
  const j = await getJSON<Journey>(
    `/analyze?model=${encodeURIComponent(model)}&bpw=${bpw}&keep_head=${keepHead}&lattice=${lattice}`,
  );
  journeyCache.set(key, j);
  return j;
}

export const probeTensor = (model: string, name: string) =>
  getJSON<TensorProbe>(`/tensor?model=${encodeURIComponent(model)}&name=${encodeURIComponent(name)}`);

// ---------------------------------------------------------------- live GPU pack

export interface LiveEvent { stage: string; msg?: string }

export async function startPack(model: string): Promise<string> {
  const r = await fetch(`${API}/pack`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
  });
  if (!r.ok) {
    const body = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(body.detail || `HTTP ${r.status}`);
  }
  return (await r.json()).job_id as string;
}

/** Subscribe to a job's SSE progress. Returns a closer; the stream ends itself on the
 *  terminal `result` event, which carries either the measured journey or an error. */
export function streamJob(
  jobId: string,
  handlers: {
    onProgress: (ev: LiveEvent) => void;
    onResult: (journey: Journey) => void;
    onError: (message: string) => void;
  },
): () => void {
  const es = new EventSource(`${API}/jobs/${jobId}/stream`);
  const close = () => es.close();

  es.addEventListener("progress", (e) => {
    try {
      handlers.onProgress(JSON.parse((e as MessageEvent).data));
    } catch {
      /* a malformed progress frame must not kill the run */
    }
  });

  es.addEventListener("result", (e) => {
    close();
    let payload: unknown;
    try {
      payload = JSON.parse((e as MessageEvent).data);
    } catch {
      handlers.onError("malformed result frame");
      return;
    }
    const err = (payload as { error?: string }).error;
    if (err) handlers.onError(err);
    else handlers.onResult(payload as Journey);
  });

  // A transport-level failure fires `error`; the server closes cleanly after `result`,
  // and close() above means this handler cannot then misreport that as a failure.
  es.onerror = () => {
    close();
    handlers.onError("lost connection to the analysis engine");
  };

  return close;
}

import { create } from "zustand";
import type { Journey, TensorProbe } from "./types";
import { analyze, cachedJourney, probeTensor, startPack, streamJob, type LiveEvent } from "./api";

export type View = "map" | "tensor" | "td3" | "journey";

/** Monotonic id of the newest /analyze request. A slower earlier response must not
 *  overwrite a newer one - dragging the bpw slider fires several in flight. */
let analyzeSeq = 0;
/** Pending debounce timer for knob-driven re-analysis. setTimeout, not rAF (banned). */
let knobTimer: ReturnType<typeof setTimeout> | null = null;
const KNOB_DEBOUNCE_MS = 250;
/** Closer for the live SSE subscription, if one is open. */
let liveClose: (() => void) | null = null;

interface State {
  model: string;
  bpw: number;
  keepHead: boolean;
  lattice: boolean;
  journey: Journey | null;
  view: View;
  selectedTensor: string | null;
  probe: TensorProbe | null;
  probeLoading: boolean;
  probeError: string | null;
  loading: boolean;
  error: string | null;
  packing: boolean;
  liveLog: LiveEvent[];
  liveError: string | null;
  setModel: (m: string) => void;
  setBpw: (b: number) => void;
  toggleKeepHead: () => void;
  toggleLattice: () => void;
  setView: (v: View) => void;
  selectTensor: (name: string) => Promise<void>;
  run: () => Promise<void>;
  runLive: () => Promise<void>;
  stopLive: () => void;
}

export const useStore = create<State>((set, get) => ({
  model: "Qwen/Qwen2.5-0.5B",
  bpw: 3.0,
  keepHead: true,
  lattice: false,
  journey: null,
  view: "map",
  selectedTensor: null,
  probe: null,
  probeLoading: false,
  probeError: null,
  loading: false,
  error: null,
  packing: false,
  liveLog: [],
  liveError: null,

  setModel: (m) => set({ model: m }),

  // Knob changes repaint immediately but coalesce the network call. A range input emits an
  // event per step, so an un-debounced run() issued one /analyze per pixel of drag.
  setBpw: (b) => {
    set({ bpw: b });
    scheduleRun(get);
  },
  toggleKeepHead: () => {
    set({ keepHead: !get().keepHead });
    scheduleRun(get);
  },
  toggleLattice: () => {
    set({ lattice: !get().lattice });
    scheduleRun(get);
  },

  setView: (v) => set({ view: v }),

  selectTensor: async (name) => {
    set({ selectedTensor: name, view: "tensor", probe: null, probeError: null, probeLoading: true });
    try {
      const p = await probeTensor(get().model, name);
      if (get().selectedTensor === name) set({ probe: p, probeLoading: false });
    } catch (e) {
      if (get().selectedTensor === name) set({ probeError: (e as Error).message, probeLoading: false });
    }
  },

  run: async () => {
    const { model, bpw, keepHead, lattice } = get();
    const m = model.trim();
    if (!m) return;
    const seq = ++analyzeSeq;
    const prevModel = get().journey?.model.name;

    // A cached journey resolves synchronously, so repeated slider positions never flash a
    // loading state.
    const hit = cachedJourney(m, bpw, keepHead, lattice);
    if (hit) {
      applyJourney(set, hit, prevModel);
      set({ loading: false, error: null });
      return;
    }

    set({ loading: true, error: null });
    try {
      const j = await analyze(m, bpw, keepHead, lattice);
      if (seq !== analyzeSeq) return;          // a newer request superseded this one
      applyJourney(set, j, prevModel);
      set({ loading: false });
    } catch (e) {
      if (seq !== analyzeSeq) return;
      set({ error: (e as Error).message, loading: false });
    }
  },

  // Measured path: enqueue a real GPU pack + eval and stream its progress. The result
  // replaces the estimated journey, which is the only way result.source becomes "measured".
  runLive: async () => {
    if (get().packing) return;
    const m = get().model.trim();
    if (!m) return;
    get().stopLive();
    set({ packing: true, liveError: null, liveLog: [], view: "journey" });
    try {
      const jobId = await startPack(m);
      liveClose = streamJob(jobId, {
        onProgress: (ev) => set({ liveLog: [...get().liveLog, ev] }),
        onResult: (j) => {
          liveClose = null;
          set({ journey: j, packing: false });
        },
        onError: (message) => {
          liveClose = null;
          set({ liveError: message, packing: false });
        },
      });
    } catch (e) {
      set({ liveError: (e as Error).message, packing: false });
    }
  },

  stopLive: () => {
    if (liveClose) {
      liveClose();
      liveClose = null;
    }
    set({ packing: false });
  },
}));

function scheduleRun(get: () => State) {
  if (knobTimer) clearTimeout(knobTimer);
  knobTimer = setTimeout(() => {
    knobTimer = null;
    void get().run();
  }, KNOB_DEBOUNCE_MS);
}

function applyJourney(
  set: (partial: Partial<State>) => void,
  j: Journey,
  prevModel: string | undefined,
) {
  // Preserve the selected tensor across same-model re-analysis (the probe is
  // bpw-independent); only reset when the model itself changed.
  const modelChanged = prevModel != null && prevModel !== j.model.name;
  set({
    journey: j,
    ...(modelChanged ? { selectedTensor: null, probe: null, probeError: null } : {}),
  });
}

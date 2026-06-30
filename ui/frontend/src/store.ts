import { create } from "zustand";
import type { Journey, TensorProbe } from "./types";
import { analyze, probeTensor } from "./api";

export type View = "map" | "tensor" | "td3" | "journey";

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
  setModel: (m: string) => void;
  setBpw: (b: number) => void;
  toggleKeepHead: () => void;
  toggleLattice: () => void;
  setView: (v: View) => void;
  selectTensor: (name: string) => Promise<void>;
  run: () => Promise<void>;
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
  setModel: (m) => set({ model: m }),
  setBpw: (b) => { set({ bpw: b }); void get().run(); },
  toggleKeepHead: () => { set({ keepHead: !get().keepHead }); void get().run(); },
  toggleLattice: () => { set({ lattice: !get().lattice }); void get().run(); },
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
    const m = get().model.trim();
    if (!m) return;
    const prevModel = get().journey?.model.name;
    set({ loading: true, error: null });
    try {
      const j = await analyze(m, get().bpw, get().keepHead, get().lattice);
      // Preserve the selected tensor across same-model re-analysis (probe is bpw-independent);
      // only reset when the model itself changed.
      const modelChanged = prevModel != null && prevModel !== j.model.name;
      set({
        journey: j,
        loading: false,
        ...(modelChanged ? { selectedTensor: null, probe: null, probeError: null } : {}),
      });
    } catch (e) {
      set({ error: (e as Error).message, loading: false });
    }
  },
}));

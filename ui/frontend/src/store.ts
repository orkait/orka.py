import { create } from "zustand";
import type { Journey } from "./types";
import { analyze } from "./api";

export type View = "map" | "tensor" | "td3" | "journey";

interface State {
  model: string;
  bpw: number;
  journey: Journey | null;
  view: View;
  selectedTensor: string | null;
  loading: boolean;
  error: string | null;
  setModel: (m: string) => void;
  setBpw: (b: number) => void;
  setView: (v: View) => void;
  selectTensor: (name: string | null) => void;
  run: () => Promise<void>;
}

export const useStore = create<State>((set, get) => ({
  model: "Qwen/Qwen2.5-0.5B",
  bpw: 3.0,
  journey: null,
  view: "map",
  selectedTensor: null,
  loading: false,
  error: null,
  setModel: (m) => set({ model: m }),
  setBpw: (b) => set({ bpw: b }),
  setView: (v) => set({ view: v }),
  selectTensor: (name) => set({ selectedTensor: name, view: "tensor" }),
  run: async () => {
    const m = get().model.trim();
    if (!m) return;
    set({ loading: true, error: null });
    try {
      const j = await analyze(m, get().bpw);
      set({ journey: j, loading: false });
    } catch (e) {
      set({ error: (e as Error).message, loading: false });
    }
  },
}));

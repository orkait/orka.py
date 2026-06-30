import { useEffect } from "react";
import { useStore } from "./store";
import { TopBar } from "./components/TopBar";
import { Rail } from "./components/Rail";
import { MapView } from "./components/views/MapView";
import { TensorView } from "./components/views/TensorView";
import { ThreeDView } from "./components/views/ThreeDView";
import { JourneyView } from "./components/views/JourneyView";

export function App() {
  const { view, journey, loading, error, run } = useStore();

  useEffect(() => {
    run();
  }, [run]);

  return (
    <div className="h-full flex flex-col">
      <TopBar />
      <div className="grid grid-cols-[264px_1fr] flex-1 min-h-0">
        <Rail />
        <main className="overflow-auto min-h-0 relative">
          {error && (
            <div className="m-6 rounded-xl border border-crit/30 bg-crit/[0.07] text-crit px-4 py-3 text-[13px]">
              {error}
            </div>
          )}
          {!journey && loading && (
            <div className="h-full flex items-center justify-center text-mut animate-pulse">analyzing model…</div>
          )}
          {!journey && !loading && !error && (
            <div className="h-full flex items-center justify-center text-dim">enter a model and press Enter</div>
          )}
          {journey && view === "map" && <MapView />}
          {journey && view === "tensor" && <TensorView />}
          {journey && view === "td3" && <ThreeDView />}
          {journey && view === "journey" && <JourneyView />}
        </main>
      </div>
    </div>
  );
}

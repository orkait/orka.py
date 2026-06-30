import { memo, useMemo } from "react";
import {
  ReactFlow, Background, BackgroundVariant, Controls, MiniMap,
  Handle, Position, type Node, type Edge, type NodeProps,
} from "@xyflow/react";
import { useStore } from "../store";
import type { ModuleEntry } from "../types";

const short = (n: string) => n.replace(/\.weight$/, "").split(".").slice(-2).join(".");
const treatClass = (t: ModuleEntry["treatment"]) =>
  t === "keep_fp16"
    ? "bg-crit/15 text-crit border-crit/30"
    : t === "skip_error_comp"
    ? "bg-warn/15 text-warn border-warn/30"
    : "bg-ac2/80 text-white border-transparent";

type IoData = { label: string; sub: string; pick: string | null; onPick: (n: string) => void };
type LayerData = { label: string; count: number; modules: ModuleEntry[]; selected: string | null; onPick: (n: string) => void };

const IoNode = memo(({ data }: NodeProps<Node<IoData>>) => (
  <div
    onClick={() => data.pick && data.onPick(data.pick)}
    className={"w-[92px] rounded-xl border px-3 py-3 text-center bg-crit/[0.07] border-crit/30 text-crit " + (data.pick ? "cursor-pointer hover:bg-crit/15" : "")}
  >
    <Handle type="target" position={Position.Left} className="!bg-bd2 !border-0" />
    <div className="text-[13px] font-semibold">{data.label}</div>
    <div className="text-[10px] text-dim mt-0.5">{data.sub}</div>
    <Handle type="source" position={Position.Right} className="!bg-bd2 !border-0" />
  </div>
));
IoNode.displayName = "IoNode";

const LayerNode = memo(({ data }: NodeProps<Node<LayerData>>) => (
  <div className="w-[190px] rounded-xl border border-bd bg-s2 p-2.5">
    {data.count > 1 && (
      <>
        <div className="absolute -top-1.5 -left-1.5 right-1.5 bottom-1.5 rounded-xl border border-bd bg-s2/60 -z-10" />
        <div className="absolute -top-0.5 -left-0.5 right-1 bottom-1 rounded-xl border border-bd bg-s2/80 -z-10" />
      </>
    )}
    <Handle type="target" position={Position.Left} className="!bg-bd2 !border-0" />
    <div className="flex items-center justify-between mb-1.5">
      <span className="text-[10px] text-dim font-semibold mono">{data.label}</span>
      {data.count > 1 && <span className="text-[10px] font-semibold rounded px-1.5 py-0.5 bg-ac/15 text-ac mono">×{data.count}</span>}
    </div>
    <div className="flex flex-col gap-1">
      {data.modules.map((m) => {
        const sel = data.selected === m.name;
        return (
          <button
            key={m.name}
            onClick={() => data.onPick(m.name)}
            title={`${m.name}  ${m.shape.join("×")}  ${m.treatment}`}
            className={"text-[10.5px] rounded-md px-2 py-1 text-left truncate border transition hover:brightness-125 " +
              treatClass(m.treatment) + (sel ? " outline outline-2 outline-ok" : "")}
          >
            {short(m.name)}
          </button>
        );
      })}
    </div>
    <Handle type="source" position={Position.Right} className="!bg-bd2 !border-0" />
  </div>
));
LayerNode.displayName = "LayerNode";

const nodeTypes = { io: IoNode, layer: LayerNode };

const COL = 240;

export function ArchFlow() {
  const journey = useStore((s) => s.journey);
  const selectedTensor = useStore((s) => s.selectedTensor);
  const selectTensor = useStore((s) => s.selectTensor);

  const { nodes, edges } = useMemo(() => {
    if (!journey) return { nodes: [] as Node[], edges: [] as Edge[] };
    const a = journey.architecture;
    const layers = a.layers.filter((l) => l.index >= 0).sort((x, y) => x.index - y.index);
    const io = a.layers.find((l) => l.index === -1)?.modules ?? [];
    const embed = io.find((m) => /embed/i.test(m.name)) ?? null;
    const head = io.find((m) => /lm_head|embed_out|output/i.test(m.name)) ?? null;

    // Collapse consecutive structurally-identical layers (same tensor-family signature) into
    // one "xN" group node. Homogeneous transformers -> embed -> [xN] -> head, not N clones.
    const sig = (l: typeof layers[number]) => l.modules.map((m) => short(m.name)).sort().join("|");
    const groups: { from: number; to: number; modules: ModuleEntry[] }[] = [];
    for (const l of layers) {
      const prev = groups[groups.length - 1];
      if (prev && sig(layers.find((x) => x.index === prev.from)!) === sig(l)) prev.to = l.index;
      else groups.push({ from: l.index, to: l.index, modules: l.modules });
    }

    const ns: Node[] = [];
    const es: Edge[] = [];
    let x = 0;
    let prev = "embed";

    ns.push({ id: "embed", type: "io", position: { x, y: 60 }, draggable: false,
      data: { label: "embed", sub: a.flags.tied_head ? "fp16 · tied" : "fp16", pick: embed?.name ?? null, onPick: selectTensor } });

    groups.forEach((g) => {
      x += COL;
      const id = `g${g.from}`;
      const count = g.to - g.from + 1;
      ns.push({ id, type: "layer", position: { x, y: 0 }, draggable: false,
        data: { label: count > 1 ? `L${g.from}–L${g.to}` : `L${g.from}`, count, modules: g.modules, selected: selectedTensor, onPick: selectTensor } });
      es.push({ id: `e-${prev}-${id}`, source: prev, target: id, type: "smoothstep",
        style: { stroke: "#2E2A3C", strokeWidth: 1.5 } });
      prev = id;
    });

    x += COL;
    ns.push({ id: "head", type: "io", position: { x, y: 60 }, draggable: false,
      data: { label: "head", sub: a.flags.tied_head ? "tied → embed" : "fp16", pick: head?.name ?? null, onPick: selectTensor } });
    es.push({ id: `e-${prev}-head`, source: prev, target: "head", type: "smoothstep",
      style: { stroke: "#2E2A3C", strokeWidth: 1.5 } });
    return { nodes: ns, edges: es };
  }, [journey, selectedTensor, selectTensor]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      colorMode="dark"
      fitView
      fitViewOptions={{ padding: 0.18 }}
      minZoom={0.12}
      maxZoom={1.6}
      nodesDraggable={false}
      nodesConnectable={false}
      elementsSelectable={false}
      edgesFocusable={false}
      panOnScroll
      proOptions={{ hideAttribution: true }}
    >
      <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="#221F2E" />
      <Controls showInteractive={false} className="!bg-s2 !border-bd" />
      <MiniMap pannable zoomable nodeColor="#2E2A3C" maskColor="rgba(8,7,13,0.7)"
        style={{ background: "#0B0A11", border: "1px solid #221F2E" }} />
    </ReactFlow>
  );
}

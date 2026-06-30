import { memo, useMemo, useState } from "react";
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

// classify a tensor by role in the transformer block's forward pass
function role(name: string): "attnIn" | "attnOut" | "mlpIn" | "mlpOut" | "other" {
  const s = short(name);
  if (/(q|k|v)_proj|query_key_value|wqkv|qkv/.test(s)) return "attnIn";
  if (/o_proj|out_proj|^.*\bwo\b|dense$/.test(s)) return "attnOut";
  if (/gate_proj|up_proj|gate_up|w1|w3/.test(s)) return "mlpIn";
  if (/down_proj|w2|c_proj/.test(s)) return "mlpOut";
  return "other";
}

type IoData = { label: string; sub: string; pick: string | null; onPick: (n: string) => void };
type LayerData = { label: string; count: number; modules: ModuleEntry[]; selected: string | null; onPick: (n: string) => void; onExpand: () => void };
type BlockData = { label: string; onCollapse: () => void };
type TensorData = { name: string; treatment: ModuleEntry["treatment"]; selected: boolean; onPick: (n: string) => void };
type OpData = { label: string };

const hStyle = "!bg-bd2 !border-0 !w-1.5 !h-1.5";

const IoNode = memo(({ data }: NodeProps<Node<IoData>>) => (
  <div onClick={() => data.pick && data.onPick(data.pick)}
    className={"nodrag nopan w-[92px] rounded-xl border px-3 py-3 text-center bg-crit/[0.07] border-crit/30 text-crit " + (data.pick ? "cursor-pointer hover:bg-crit/15" : "")}>
    <Handle type="target" position={Position.Left} className={hStyle} />
    <div className="text-[13px] font-semibold">{data.label}</div>
    <div className="text-[10px] text-dim mt-0.5">{data.sub}</div>
    <Handle type="source" position={Position.Right} className={hStyle} />
  </div>
));
IoNode.displayName = "IoNode";

const LayerNode = memo(({ data }: NodeProps<Node<LayerData>>) => (
  <div className="w-[190px] rounded-xl border border-bd bg-s2 p-2.5">
    {data.count > 1 && <>
      <div className="absolute -top-1.5 -left-1.5 right-1.5 bottom-1.5 rounded-xl border border-bd bg-s2/60 -z-10" />
      <div className="absolute -top-0.5 -left-0.5 right-1 bottom-1 rounded-xl border border-bd bg-s2/80 -z-10" />
    </>}
    <Handle type="target" position={Position.Left} className={hStyle} />
    <div className="flex items-center justify-between mb-1.5">
      <span className="text-[10px] text-dim font-semibold mono">{data.label}</span>
      <div className="flex items-center gap-1">
        {data.count > 1 && <span className="text-[10px] font-semibold rounded px-1.5 py-0.5 bg-ac/15 text-ac mono">×{data.count}</span>}
        <button onClick={data.onExpand} title="expand block dataflow"
          className="nodrag nopan text-ac text-[12px] leading-none rounded px-1 hover:bg-ac/15">⤢</button>
      </div>
    </div>
    <div className="flex flex-col gap-1">
      {data.modules.map((m) => {
        const sel = data.selected === m.name;
        return (
          <button key={m.name} onClick={() => data.onPick(m.name)}
            title={`${m.name}  ${m.shape.join("×")}  ${m.treatment}`}
            className={"nodrag nopan text-[10.5px] rounded-md px-2 py-1 text-left truncate border transition hover:brightness-125 " +
              treatClass(m.treatment) + (sel ? " outline outline-2 outline-ok" : "")}>
            {short(m.name)}
          </button>
        );
      })}
    </div>
    <Handle type="source" position={Position.Right} className={hStyle} />
  </div>
));
LayerNode.displayName = "LayerNode";

const BlockNode = memo(({ data }: NodeProps<Node<BlockData>>) => (
  <div className="w-full h-full rounded-2xl border border-ac/40 bg-s3/60"
    style={{ boxShadow: "0 0 40px -10px rgba(139,124,246,.3)" }}>
    <Handle type="target" position={Position.Left} className={hStyle} />
    <div className="flex items-center justify-between px-3 py-2 border-b border-bd">
      <span className="text-[11px] font-semibold text-ac mono">{data.label} · forward dataflow</span>
      <button onClick={data.onCollapse} title="collapse" className="nodrag nopan text-mut text-[13px] leading-none rounded px-1 hover:bg-bd hover:text-tx">✕</button>
    </div>
    <Handle type="source" position={Position.Right} className={hStyle} />
  </div>
));
BlockNode.displayName = "BlockNode";

const TensorNode = memo(({ data }: NodeProps<Node<TensorData>>) => (
  <div onClick={() => data.onPick(data.name)} title={data.name}
    className={"nodrag nopan w-[112px] rounded-lg border px-2 py-1.5 text-[10px] text-center cursor-pointer transition hover:brightness-125 " +
      treatClass(data.treatment) + (data.selected ? " outline outline-2 outline-ok" : "")}>
    <Handle type="target" position={Position.Left} className={hStyle} />
    {short(data.name)}
    <Handle type="source" position={Position.Right} className={hStyle} />
  </div>
));
TensorNode.displayName = "TensorNode";

const OpNode = memo(({ data }: NodeProps<Node<OpData>>) => (
  <div className="w-[64px] rounded-full border border-bd2 bg-bg/80 px-2 py-1.5 text-[10px] text-center text-mut mono">
    <Handle type="target" position={Position.Left} className={hStyle} />
    {data.label}
    <Handle type="source" position={Position.Right} className={hStyle} />
  </div>
));
OpNode.displayName = "OpNode";

const nodeTypes = { io: IoNode, layer: LayerNode, block: BlockNode, tensor: TensorNode, op: OpNode };

const GAP = 60;
const COLW = 210;     // collapsed group width
const BLOCKW = 860;   // expanded block width
const BLOCKH = 250;
const flow = { stroke: "#2E2A3C", strokeWidth: 1.5 };
const dataEdge = (id: string, s: string, t: string, animated = false): Edge =>
  ({ id, source: s, target: t, type: "smoothstep", animated, style: { stroke: "#5B54FF", strokeWidth: 1.5 } });

// lay out one layer's tensors as the block forward graph; returns child nodes + intra edges
function blockChildren(blockId: string, modules: ModuleEntry[], selected: string | null, onPick: (n: string) => void) {
  const by: Record<string, ModuleEntry[]> = { attnIn: [], attnOut: [], mlpIn: [], mlpOut: [], other: [] };
  for (const m of modules) by[role(m.name)].push(m);
  const ns: Node[] = [];
  const es: Edge[] = [];
  const child = (m: ModuleEntry, x: number, y: number) => {
    ns.push({ id: `${blockId}/${m.name}`, type: "tensor", parentId: blockId, extent: "parent", position: { x, y },
      data: { name: m.name, treatment: m.treatment, selected: selected === m.name, onPick } });
    return `${blockId}/${m.name}`;
  };
  const op = (key: string, label: string, x: number, y: number) => {
    ns.push({ id: `${blockId}#${key}`, type: "op", parentId: blockId, extent: "parent", position: { x, y }, data: { label } });
    return `${blockId}#${key}`;
  };

  const aIn = by.attnIn.map((m, i) => child(m, 16, 52 + i * 46));
  const attn = op("attn", "attn", 156, 98);
  const aOut = by.attnOut.map((m) => child(m, 268, 98));
  const mIn = by.mlpIn.map((m, i) => child(m, 432, 70 + i * 52));
  const mlp = op("mlp", "mlp", 576, 98);
  const mOut = by.mlpOut.map((m) => child(m, 700, 98));
  by.other.forEach((m, i) => child(m, 16 + i * 120, 196));

  aIn.forEach((s) => es.push(dataEdge(`${s}->attn`, s, attn, true)));
  aOut.forEach((t) => es.push(dataEdge(`attn->${t}`, attn, t, true)));
  const afterAttn = aOut[0] ?? attn;
  mIn.forEach((t) => es.push(dataEdge(`${afterAttn}->${t}`, afterAttn, t)));
  mIn.forEach((s) => es.push(dataEdge(`${s}->mlp`, s, mlp, true)));
  mOut.forEach((t) => es.push(dataEdge(`mlp->${t}`, mlp, t, true)));
  return { ns, es, entry: aIn[0] ?? attn, exit: mOut[0] ?? mlp };
}

export function ArchFlow() {
  const journey = useStore((s) => s.journey);
  const selectedTensor = useStore((s) => s.selectedTensor);
  const selectTensor = useStore((s) => s.selectTensor);
  const [expanded, setExpanded] = useState<number | null>(null);

  const { nodes, edges } = useMemo(() => {
    if (!journey) return { nodes: [] as Node[], edges: [] as Edge[] };
    const a = journey.architecture;
    const layers = a.layers.filter((l) => l.index >= 0).sort((x, y) => x.index - y.index);
    const io = a.layers.find((l) => l.index === -1)?.modules ?? [];
    const embed = io.find((m) => /embed/i.test(m.name)) ?? null;
    const head = io.find((m) => /lm_head|embed_out|output/i.test(m.name)) ?? null;

    const sig = (l: typeof layers[number]) => l.modules.map((m) => short(m.name)).sort().join("|");
    const groups: { from: number; to: number; modules: ModuleEntry[] }[] = [];
    for (const l of layers) {
      const p = groups[groups.length - 1];
      if (p && sig(layers.find((x) => x.index === p.from)!) === sig(l)) p.to = l.index;
      else groups.push({ from: l.index, to: l.index, modules: l.modules });
    }

    const ns: Node[] = [];
    const es: Edge[] = [];
    let x = 0;
    let prev = "embed";
    let prevW = 92;
    const link = (target: string) => es.push({ id: `e-${prev}-${target}`, source: prev, target, type: "smoothstep", style: flow });

    ns.push({ id: "embed", type: "io", position: { x, y: BLOCKH / 2 - 30 }, draggable: false,
      data: { label: "embed", sub: a.flags.tied_head ? "fp16 · tied" : "fp16", pick: embed?.name ?? null, onPick: selectTensor } });

    for (const g of groups) {
      x += GAP + prevW;
      const id = `g${g.from}`;
      const count = g.to - g.from + 1;
      if (expanded === g.from) {
        ns.push({ id, type: "block", position: { x, y: 0 }, draggable: false,
          style: { width: BLOCKW, height: BLOCKH },
          data: { label: count > 1 ? `L${g.from}–L${g.to} (rep. L${g.from})` : `L${g.from}`, onCollapse: () => setExpanded(null) } });
        const { ns: cn, es: ce } = blockChildren(id, g.modules, selectedTensor, selectTensor);
        ns.push(...cn); es.push(...ce);
        prevW = BLOCKW;
      } else {
        ns.push({ id, type: "layer", position: { x, y: BLOCKH / 2 - 90 }, draggable: false, style: { width: COLW },
          data: { label: count > 1 ? `L${g.from}–L${g.to}` : `L${g.from}`, count, modules: g.modules, selected: selectedTensor, onPick: selectTensor, onExpand: () => setExpanded(g.from) } });
        prevW = COLW;
      }
      link(id);
      prev = id;
    }

    x += GAP + prevW;
    ns.push({ id: "head", type: "io", position: { x, y: BLOCKH / 2 - 30 }, draggable: false,
      data: { label: "head", sub: a.flags.tied_head ? "tied → embed" : "fp16", pick: head?.name ?? null, onPick: selectTensor } });
    link("head");
    return { nodes: ns, edges: es };
  }, [journey, selectedTensor, selectTensor, expanded]);

  return (
    <ReactFlow
      nodes={nodes} edges={edges} nodeTypes={nodeTypes}
      colorMode="dark" fitView fitViewOptions={{ padding: 0.18 }}
      minZoom={0.1} maxZoom={1.8}
      nodesDraggable={false} nodesConnectable={false} elementsSelectable={false} edgesFocusable={false}
      zoomOnScroll panOnDrag zIndexMode="auto"
      proOptions={{ hideAttribution: true }}
    >
      <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="#221F2E" />
      <Controls showInteractive={false} className="!bg-s2 !border-bd" />
      <MiniMap pannable zoomable nodeColor="#2E2A3C" maskColor="rgba(8,7,13,0.7)"
        style={{ background: "#0B0A11", border: "1px solid #221F2E" }} />
    </ReactFlow>
  );
}

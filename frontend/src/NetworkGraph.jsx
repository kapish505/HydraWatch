import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'

/**
 * NetworkGraph — D3.js network visualization with live pressure data.
 *
 * Nodes = junctions (where sensors are)
 * Edges = pipes connecting them
 *
 * Node colors reflect LIVE state:
 *   Blue (#378ADD)  — normal pressure, no anomaly
 *   Amber (#BA7517) — GAT probability > 0.3 (elevated risk)
 *   Red (#E24B4A)   — suspect node (active alert)
 *
 * Node SIZE scales with current pressure.
 * Suspect nodes get a pulsing red ring animation.
 * Hover tooltip shows pressure, GAT probability, and node type.
 */
export default function NetworkGraph({ data, currentData }) {
  const containerRef = useRef(null)
  const svgRef = useRef(null)
  const gRef = useRef(null)
  const nodesRef = useRef(null)
  const nodeMapRef = useRef(null)
  const [tooltip, setTooltip] = useState(null)

  // ── Initial render ──────────────────────────────────────────────
  useEffect(() => {
    if (!data || !containerRef.current) return

    const container = containerRef.current
    const width = container.clientWidth
    const height = container.clientHeight

    // Clear previous
    d3.select(container).selectAll('svg').remove()

    const svg = d3.select(container)
      .append('svg')
      .attr('width', '100%')
      .attr('height', '100%')
      .attr('viewBox', `0 0 ${width} ${height}`)
      .call(d3.zoom().scaleExtent([0.3, 5]).on('zoom', (e) => {
        g.attr('transform', e.transform)
      }))

    const g = svg.append('g')
    svgRef.current = svg
    gRef.current = g

    // Scale coordinates to fit viewport
    const nodesWithCoords = data.nodes.filter(n => n.x !== undefined)
    if (nodesWithCoords.length === 0) return

    const xExtent = d3.extent(nodesWithCoords, d => d.x)
    const yExtent = d3.extent(nodesWithCoords, d => d.y)

    const padding = 60
    const scaleX = d3.scaleLinear().domain(xExtent).range([padding, width - padding])
    const scaleY = d3.scaleLinear().domain(yExtent).range([height - padding, padding])

    // Assign pixel positions
    const nodeMap = new Map()
    data.nodes.forEach(n => {
      const px = n.x !== undefined ? scaleX(n.x) : width / 2 + (Math.random() - 0.5) * 200
      const py = n.y !== undefined ? scaleY(n.y) : height / 2 + (Math.random() - 0.5) * 200
      nodeMap.set(n.id, { ...n, px, py })
    })
    nodeMapRef.current = nodeMap

    // Draw edges (pipes)
    g.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(data.edges)
      .enter()
      .append('line')
      .attr('class', 'link')
      .attr('x1', d => nodeMap.get(d.source)?.px || 0)
      .attr('y1', d => nodeMap.get(d.source)?.py || 0)
      .attr('x2', d => nodeMap.get(d.target)?.px || 0)
      .attr('y2', d => nodeMap.get(d.target)?.py || 0)
      .attr('stroke-width', d => Math.max(1, (d.diameter || 0) * 8))

    // Draw nodes (junctions)
    const nodes = g.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(Array.from(nodeMap.values()))
      .enter()
      .append('g')
      .attr('class', 'node')
      .attr('transform', d => `translate(${d.px}, ${d.py})`)
      .attr('data-id', d => d.id)

    // Pulse circle (hidden by default, shown on alert)
    nodes.append('circle')
      .attr('class', 'pulse-ring')
      .attr('r', 16)
      .attr('fill', 'none')
      .attr('stroke', '#E24B4A')
      .attr('stroke-width', 2)
      .attr('opacity', 0)

    // Main circle
    nodes.append('circle')
      .attr('class', 'main-circle')
      .attr('r', d => d.type === 'Reservoir' ? 10 : 7)
      .attr('fill', d => d.type === 'Reservoir' ? '#06B6D4' : '#378ADD')
      .attr('stroke', 'rgba(255,255,255,0.15)')
      .attr('stroke-width', 1.5)

    // Pressure value label (small, below node)
    nodes.append('text')
      .attr('class', 'pressure-label')
      .attr('dy', 18)
      .attr('text-anchor', 'middle')
      .style('font-family', 'JetBrains Mono, monospace')
      .style('font-size', '8px')
      .style('fill', '#64748B')
      .style('pointer-events', 'none')
      .text('')

    // Node ID label
    nodes.append('text')
      .attr('dx', 12)
      .attr('dy', 3)
      .text(d => d.id.replace('Node_', 'N'))

    nodesRef.current = nodes

    // Mouse events
    nodes.on('mouseenter', function(event, d) {
      const [x, y] = d3.pointer(event, container)
      setTooltip({ x, y, node: d })
    })
    .on('mouseleave', () => setTooltip(null))

  }, [data])

  // ── Update node colors + pressure labels based on live data ────
  useEffect(() => {
    if (!nodesRef.current || !currentData) return

    const suspectNodes = new Set(
      currentData.active_alert?.suspect_nodes || currentData.suspect_nodes || []
    )
    const predictions = currentData.predictions || {}
    const pressures = currentData.pressures || {}

    // Update main circle color based on GAT predictions
    nodesRef.current.selectAll('.main-circle')
      .transition()
      .duration(200)
      .attr('fill', function(d) {
        if (d.type === 'Reservoir') return '#06B6D4'
        if (suspectNodes.has(d.id)) return '#E24B4A'
        const prob = predictions[d.id] || 0
        if (prob > 0.5) return '#E24B4A'
        if (prob > 0.3) return '#BA7517'
        return '#378ADD'
      })
      .attr('r', function(d) {
        if (d.type === 'Reservoir') return 10
        if (suspectNodes.has(d.id)) return 11
        const prob = predictions[d.id] || 0
        return prob > 0.3 ? 9 : 7
      })
      .attr('stroke', function(d) {
        if (suspectNodes.has(d.id)) return '#E24B4A'
        const prob = predictions[d.id] || 0
        if (prob > 0.3) return '#BA7517'
        return 'rgba(255,255,255,0.15)'
      })
      .attr('stroke-width', function(d) {
        if (suspectNodes.has(d.id)) return 3
        return 1.5
      })

    // Update pressure value labels
    nodesRef.current.selectAll('.pressure-label')
      .text(function(d) {
        const p = pressures[d.id]
        return p !== undefined ? p.toFixed(1) : ''
      })
      .style('fill', function(d) {
        if (suspectNodes.has(d.id)) return '#E24B4A'
        const prob = predictions[d.id] || 0
        if (prob > 0.3) return '#BA7517'
        return '#64748B'
      })

    // Pulse animation on suspect nodes
    nodesRef.current.selectAll('.pulse-ring')
      .attr('opacity', function(d) {
        return suspectNodes.has(d.id) ? 0.6 : 0
      })

    if (suspectNodes.size > 0) {
      nodesRef.current.selectAll('.pulse-ring')
        .filter(d => suspectNodes.has(d.id))
        .transition()
        .duration(1000)
        .attr('r', 22)
        .attr('opacity', 0)
        .transition()
        .duration(0)
        .attr('r', 10)
        .attr('opacity', 0.6)
    }

  }, [currentData])

  return (
    <div ref={containerRef} className="w-full h-full relative cursor-move">
      {/* Legend */}
      <div className="absolute top-3 left-3 z-10 flex gap-2">
        <div className="bg-black/50 backdrop-blur px-2.5 py-1 rounded-lg border border-white/5 text-[10px] text-hw-muted flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-[#378ADD]" /> Normal
        </div>
        <div className="bg-black/50 backdrop-blur px-2.5 py-1 rounded-lg border border-white/5 text-[10px] text-hw-muted flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-[#BA7517]" /> Warning
        </div>
        <div className="bg-black/50 backdrop-blur px-2.5 py-1 rounded-lg border border-white/5 text-[10px] text-hw-muted flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-[#E24B4A] shadow-[0_0_6px_rgba(226,75,74,0.6)]" /> Alert
        </div>
      </div>

      {/* Tooltip */}
      {tooltip && (
        <div
          className="absolute bg-black/90 backdrop-blur-md border border-white/10 rounded-lg p-3 text-xs w-48 z-20 shadow-xl pointer-events-none"
          style={{ left: tooltip.x + 15, top: tooltip.y + 15 }}
        >
          <div className="font-bold text-white mb-2 pb-1 border-b border-white/10">
            {tooltip.node.id}
          </div>
          <div className="space-y-1">
            <div className="flex justify-between">
              <span className="text-hw-muted">Pressure:</span>
              <span className="text-white font-mono">
                {currentData?.pressures?.[tooltip.node.id]?.toFixed(2) || '--'} m
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-hw-muted">Type:</span>
              <span className="text-white">{tooltip.node.type}</span>
            </div>
            {currentData?.predictions?.[tooltip.node.id] !== undefined && (
              <div className="flex justify-between">
                <span className="text-hw-muted">GAT Prob:</span>
                <span className={`font-mono ${
                  currentData.predictions[tooltip.node.id] > 0.5 ? 'text-hw-red' :
                  currentData.predictions[tooltip.node.id] > 0.3 ? 'text-hw-amber' : 'text-hw-green'
                }`}>
                  {(currentData.predictions[tooltip.node.id] * 100).toFixed(1)}%
                </span>
              </div>
            )}
            {currentData?.active_alert?.suspect_nodes?.includes(tooltip.node.id) && (
              <div className="mt-1 pt-1 border-t border-white/10 text-hw-red font-bold">
                ⚠ SUSPECT NODE
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

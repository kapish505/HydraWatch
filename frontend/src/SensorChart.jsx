import { useRef, useEffect, useCallback } from 'react'
import { Chart, registerables } from 'chart.js'

Chart.register(...registerables)

/**
 * SensorChart — Chart.js time series of pressure readings.
 *
 * Shows the last 60 timesteps with one line per sensor node.
 * Y-axis = pressure in metres.
 * Updates incrementally instead of destroying/recreating.
 */
export default function SensorChart({ history, nodeNames }) {
  const canvasRef = useRef(null)
  const chartRef = useRef(null)

  // Color palette for sensor lines
  const colorsRef = useRef([
    '#378ADD', '#06B6D4', '#10B981', '#8B5CF6',
    '#F59E0B', '#EC4899', '#14B8A6', '#6366F1',
  ])

  // Create or update chart
  useEffect(() => {
    if (!canvasRef.current || history.length === 0 || nodeNames.length === 0) return

    const colors = colorsRef.current
    const labels = history.map(d => {
      const ts = d.timestamp || `T${d.timestep}`
      return typeof ts === 'string' && ts.length > 10 ? ts.slice(11, 16) : `${d.timestep}`
    })

    const datasets = nodeNames.map((node, idx) => ({
      label: node.replace('Node_', 'N'),
      data: history.map(d => d.pressures?.[node] ?? null),
      borderColor: colors[idx % colors.length],
      backgroundColor: 'transparent',
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.3,
      fill: false,
    }))

    // If chart exists, update it in-place (much faster, no flicker)
    if (chartRef.current) {
      chartRef.current.data.labels = labels
      chartRef.current.data.datasets = datasets
      chartRef.current.update('none') // 'none' = no animation for speed
      return
    }

    // First time: create chart
    const ctx = canvasRef.current.getContext('2d')
    chartRef.current = new Chart(ctx, {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: {
          mode: 'nearest',
          intersect: false,
        },
        layout: {
          padding: { top: 5, bottom: 0, left: 0, right: 5 },
        },
        plugins: {
          legend: {
            display: true,
            position: 'top',
            align: 'start',
            labels: {
              color: '#94A3B8',
              font: { size: 10, family: 'JetBrains Mono' },
              boxWidth: 8,
              boxHeight: 8,
              padding: 8,
              usePointStyle: true,
            },
          },
          tooltip: {
            backgroundColor: 'rgba(0,0,0,0.85)',
            titleFont: { family: 'JetBrains Mono', size: 11 },
            bodyFont: { family: 'Inter', size: 11 },
            padding: 10,
            cornerRadius: 8,
          },
        },
        scales: {
          x: {
            display: true,
            grid: { color: 'rgba(255,255,255,0.03)' },
            ticks: {
              color: '#64748B',
              font: { size: 9, family: 'JetBrains Mono' },
              maxRotation: 0,
              maxTicksLimit: 8,
            },
          },
          y: {
            display: true,
            title: {
              display: true,
              text: 'Pressure (m)',
              color: '#94A3B8',
              font: { size: 10 },
            },
            grid: { color: 'rgba(255,255,255,0.03)' },
            ticks: {
              color: '#64748B',
              font: { size: 9, family: 'JetBrains Mono' },
            },
          },
        },
      },
    })

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy()
        chartRef.current = null
      }
    }
  }, [history, nodeNames])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (chartRef.current) {
        chartRef.current.destroy()
        chartRef.current = null
      }
    }
  }, [])

  if (history.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center text-hw-muted text-sm">
        <div className="text-center">
          <div className="text-2xl mb-2 opacity-30">📊</div>
          <div>Start a simulation to see pressure readings</div>
        </div>
      </div>
    )
  }

  return (
    <div className="w-full h-full relative">
      <canvas ref={canvasRef} />
    </div>
  )
}

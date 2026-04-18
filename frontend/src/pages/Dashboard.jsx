import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useAuth } from '../contexts/AuthContext'
import { API_BASE, WS_BASE } from '../config'
import NetworkGraph from '../NetworkGraph'
import SensorChart from '../SensorChart'

export default function Dashboard() {
  const { logout } = useAuth()
  
  // ── State ──────────────────────────────────────────────────────
  const [networkData, setNetworkData] = useState(null)
  const [currentData, setCurrentData] = useState(null)
  const [alerts, setAlerts] = useState([])
  const [pressureHistory, setPressureHistory] = useState([])
  const [isPlaying, setIsPlaying] = useState(false)
  const [scenarioId, setScenarioId] = useState(1)
  const [speed, setSpeed] = useState(300)
  const [scenarios, setScenarios] = useState([])
  const [systemStatus, setSystemStatus] = useState('normal') // normal, warning, alert
  const [lastShapAlert, setLastShapAlert] = useState(null)
  
  const wsRef = useRef(null)
  const lastAlertKeyRef = useRef(null)

  // ── Load network & scenarios ───────────────────────────────────
  useEffect(() => {
    fetch(`${API_BASE}/network`)
      .then(res => res.json())
      .then(data => setNetworkData(data))
      .catch(err => console.error('Network load failed:', err))

    fetch(`${API_BASE}/api/scenarios`)
      .then(res => res.json())
      .then(data => setScenarios(data.scenarios || []))
      .catch(err => console.error('Scenarios load failed:', err))
  }, [])

  // ── WebSocket handling ─────────────────────────────────────────
  const startSimulation = useCallback(() => {
    if (wsRef.current) wsRef.current.close()

    const ws = new WebSocket(`${WS_BASE}/ws/stream`)
    wsRef.current = ws
    lastAlertKeyRef.current = null

    ws.onopen = () => {
      setIsPlaying(true)
      setSystemStatus('normal')
      setPressureHistory([])
      setAlerts([])
      setLastShapAlert(null)
      ws.send(JSON.stringify({ scenario_id: scenarioId, speed }))
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)

      if (data.done) {
        setIsPlaying(false)
        return
      }

      setCurrentData(data)

      // Keep last 60 timesteps
      setPressureHistory(prev => {
        const next = [...prev, data]
        return next.length > 60 ? next.slice(-60) : next
      })

      // Dedup Alert Feed
      if (data.active_alert) {
        const alertKey = (data.active_alert.suspect_nodes || []).sort().join(',')
        setSystemStatus('alert')
        setLastShapAlert(data.active_alert) // Keep SHAP active between ticks

        if (alertKey !== lastAlertKeyRef.current) {
          lastAlertKeyRef.current = alertKey
          setAlerts(prev => [{
            id: Date.now(),
            timestamp: data.timestamp,
            ...data.active_alert,
          }, ...prev].slice(0, 20))
        }
      }

      // Process SHAP features on every timestep if available
      if (data.shap_features && data.shap_features.length > 0) {
        setLastShapAlert(prev => ({
          ...(prev || {}),
          shap_features: data.shap_features,
          suspect_nodes: data.suspect_nodes || prev?.suspect_nodes || [],
        }))
      }

      if (data.prediction === 1 || data.xgb_probability > 0.3) {
        setSystemStatus('warning')
      } else {
        setSystemStatus('normal')
        lastAlertKeyRef.current = null
      }
    }

    ws.onclose = () => setIsPlaying(false)
    ws.onerror = () => setIsPlaying(false)
  }, [scenarioId, speed])

  const stopSimulation = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setIsPlaying(false)
  }, [])

  // ── Derived Variables ──────────────────────────────────────────
  const probability = currentData?.xgb_probability || 0
  const timestep = currentData?.timestep || 0
  const isAlert = currentData?.active_alert != null
  const alertColor = isAlert ? 'text-error' : probability > 0.3 ? 'text-amber-500' : 'text-primary'
  
  // Format SHAP features
  const shapFeatures = (lastShapAlert?.shap_features || []).sort((a, b) => Math.abs(b.impact || 0) - Math.abs(a.impact || 0)).slice(0, 5)
  const maxShap = shapFeatures.length > 0 ? Math.max(...shapFeatures.map(f => Math.abs(f.impact || 0)), 0.01) : 1

  return (
    <div className="h-screen w-full flex flex-col font-['Inter'] bg-[#11131b] text-[#e1e2ed] overflow-hidden">
      {/* ── HEADER ── */}
      <header className="fixed top-0 w-full z-50 bg-[#11131b]/80 backdrop-blur-xl border-b border-[#3d494c]/20 shadow-[0_0_20px_rgba(6,182,212,0.1)] flex justify-between items-center px-8 h-16">
        <div className="flex items-center gap-6">
          <span className="text-xl font-bold tracking-tighter text-[#4cd7f6] uppercase font-headline">HydraWatch</span>
          <div className="h-6 w-px bg-outline-variant/30"></div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${
                systemStatus === 'alert' ? 'bg-error shadow-[0_0_10px_rgba(255,180,171,0.8)] animate-pulse' :
                systemStatus === 'warning' ? 'bg-amber-500 shadow-[0_0_10px_rgba(245,158,11,0.5)]' :
                'bg-tertiary status-pulse'
              }`}></span>
              <span className={`mono text-[10px] tracking-widest uppercase ${
                systemStatus === 'alert' ? 'text-error font-bold' :
                systemStatus === 'warning' ? 'text-amber-500 font-bold' :
                'text-tertiary'
              }`}>
                {systemStatus === 'alert' ? 'System: Critical Alert' :
                 systemStatus === 'warning' ? 'System: Elevated Risk' :
                 'System: Nominal'}
              </span>
            </div>
            <span className="mono text-xs text-slate-500">|</span>
            <span className="mono text-[10px] text-on-surface-variant uppercase tracking-tighter">
              T={String(timestep).padStart(3, '0')}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-6">
          <div className="mono text-sm text-primary font-bold">
            {currentData?.timestamp ? currentData.timestamp.split(' ')[1] : '00:00:00'}
          </div>
          <div className="flex items-center gap-4 text-slate-400">
            <button onClick={logout} className="material-symbols-outlined hover:text-error cursor-pointer transition-colors" title="Logout">power_settings_new</button>
          </div>
        </div>
      </header>

      {/* ── SIDEBAR ── */}
      <aside className="h-screen w-16 fixed left-0 top-0 bg-[#0c0e15] border-r border-[#3d494c]/10 flex flex-col items-center py-20 z-40">
        <div className="flex flex-col gap-8">
          <span className="material-symbols-outlined text-primary" style={{fontVariationSettings: "'FILL' 1"}}>water_drop</span>
          <span className="material-symbols-outlined text-slate-600 hover:text-white cursor-pointer transition-colors">dashboard</span>
          <span className="material-symbols-outlined text-slate-600 hover:text-white cursor-pointer transition-colors">sensors</span>
          <span className="material-symbols-outlined text-slate-600 hover:text-white cursor-pointer transition-colors">psychology</span>
        </div>
        <div className="mt-auto mb-8 flex flex-col gap-6">
          <span className="material-symbols-outlined text-slate-700 hover:text-white cursor-pointer text-sm transition-colors">terminal</span>
        </div>
      </aside>

      {/* ── MAIN WORKSPACE ── */}
      <main className="flex-1 ml-16 mt-16 p-4 grid grid-cols-1 md:grid-cols-12 gap-4 h-[calc(100vh-64px)] overflow-hidden">
        
        {/* LEFT COLUMN: Controls & Analytics */}
        <section className="col-span-1 md:col-span-3 flex flex-col gap-4 overflow-y-auto pr-1">
          
          {/* Controls */}
          <motion.div initial={{opacity:0, x:-20}} animate={{opacity:1, x:0}} className="bg-surface-container rounded-xl p-5 border border-outline-variant/10 shadow-lg shrink-0">
            <div className="flex justify-between items-center mb-6">
              <h3 className="font-bold text-[10px] uppercase tracking-[0.2em] text-on-surface-variant">Simulation Target</h3>
              <span className="material-symbols-outlined text-primary text-sm">tune</span>
            </div>
            <div className="space-y-4">
              <select 
                value={scenarioId} onChange={e => setScenarioId(Number(e.target.value))}
                className="w-full bg-surface-container-lowest border-none text-sm text-on-surface rounded-md focus:ring-1 focus:ring-primary py-2 px-3"
              >
                {scenarios.length > 0 ? scenarios.map(s => (
                  <option key={s.id} value={s.id}>Scenario {s.id} — Node {s.leak_node}</option>
                )) : [...Array(10)].map((_, i) => <option key={i+1} value={i+1}>Hanoi Baseline {i+1}</option>)}
              </select>

              <div className="flex flex-col gap-2 pt-2">
                 <div className="flex justify-between items-center px-1">
                    <span className="text-[10px] uppercase font-mono text-slate-500">Polling</span>
                    <span className="mono text-[10px] text-primary">{speed}ms</span>
                 </div>
                 <input 
                   type="range" min="50" max="1000" step="50" value={speed} onChange={e => setSpeed(Number(e.target.value))}
                   className="w-full h-1 bg-surface-container-highest rounded-full appearance-none accent-primary" 
                 />
              </div>

              <div className="grid grid-cols-2 gap-2 mt-4">
                <button 
                  onClick={startSimulation} disabled={isPlaying}
                  className="bg-primary/10 text-primary border border-primary/20 py-2 rounded text-[10px] font-bold uppercase tracking-widest hover:bg-primary hover:text-on-primary-container transition-all disabled:opacity-30 flex items-center justify-center gap-1"
                >
                  <span className="material-symbols-outlined text-base">play_arrow</span> Start
                </button>
                <button 
                  onClick={stopSimulation} disabled={!isPlaying}
                  className="bg-surface-container-high py-2 rounded text-[10px] font-bold uppercase tracking-widest text-on-surface hover:bg-surface-bright transition-colors disabled:opacity-30 flex items-center justify-center gap-1"
                >
                  <span className="material-symbols-outlined text-base">stop</span> Halt
                </button>
              </div>
            </div>
          </motion.div>

          {/* Probability HUD */}
          <motion.div initial={{opacity:0, scale:0.95}} animate={{opacity:1, scale:1}} className="bg-surface-container rounded-xl p-5 border border-outline-variant/10 flex flex-col justify-center items-center relative overflow-hidden shrink-0 min-h-[140px]">
            {isAlert && <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-error to-transparent animate-pulse delay-75"></div>}
            {systemStatus === 'warning' && <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-amber-500 to-transparent"></div>}
            
            <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-on-surface-variant mb-2 z-10">Real-time XGBoost Probability</h3>
            <div className={`mono text-6xl font-black leading-none mb-1 z-10 transition-colors duration-500 ${alertColor}`}>
              {(probability * 100).toFixed(1)}<span className="text-2xl font-normal opacity-50">%</span>
            </div>
            {currentData?.label > 0 && (
              <motion.div initial={{opacity:0}} animate={{opacity:1}} className="absolute bottom-2 bg-error/20 px-3 py-0.5 rounded text-[9px] text-error font-bold tracking-widest mono border border-error/20">
                 GROUND TRUTH LEAK ACTIVE
              </motion.div>
            )}
          </motion.div>

          {/* TreeSHAP Matrix */}
          <motion.div initial={{opacity:0, x:-20}} animate={{opacity:1, x:0}} className="bg-surface-container rounded-xl p-5 border border-outline-variant/10 flex-1 flex flex-col min-h-0">
            <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-on-surface-variant mb-4 flex items-center gap-2 shrink-0">
              <span className="material-symbols-outlined text-sm text-primary">account_tree</span>
              Live Feature Impact
            </h3>
            <div className="flex-1 space-y-4 overflow-y-auto pr-1">
              {shapFeatures.length > 0 ? (
                <AnimatePresence>
                  {shapFeatures.map((feat, i) => {
                    const pct = Math.min((Math.abs(feat.impact) / maxShap) * 100, 100)
                    const isPositive = feat.impact > 0
                    const fName = (() => {
                      const raw = feat.feature
                      // Extract node number and feature type
                      const m = raw.match(/^Node_(\d+)_(.+)$/)
                      if (!m) return raw.replace(/_/g, ' ')
                      const node = m[1]
                      const suffix = m[2]
                      const labels = {
                        'rolling_mean_1h': '1h Avg Pressure',
                        'rolling_std_1h': '1h Pressure Volatility',
                        'rolling_mean_6h': '6h Avg Pressure',
                        'rolling_std_6h': '6h Pressure Volatility',
                        'pressure_z_score': 'Pressure Anomaly Score',
                        'pressure_gradient': 'Pressure Rate of Change',
                        'neighbor_pressure_delta': 'Neighbor Pressure Gap',
                        'demand_residual': 'Demand Deviation',
                        'hour_of_day': 'Time of Day',
                        'day_of_week': 'Day of Week',
                        'pressure_range_6h': '6h Pressure Range',
                        'cumulative_deviation': 'Cumulative Drift',
                        'neighbor_z_diff': 'Neighbor Anomaly Gap',
                      }
                      return `Node ${node}: ${labels[suffix] || suffix.replace(/_/g, ' ')}`
                    })()
                    
                    return (
                      <motion.div layout initial={{opacity:0, x:-10}} animate={{opacity:1, x:0}} key={feat.feature} className="space-y-1 w-full">
                        <div className="flex justify-between text-[10px] text-slate-400 font-medium tracking-tight">
                           <span className="truncate pr-2">{fName}</span>
                           <span className={`mono ${isPositive ? 'text-primary' : 'text-tertiary'}`}>
                             {isPositive ? '+' : ''}{feat.impact.toFixed(3)}
                           </span>
                        </div>
                        <div className={`w-full h-1.5 bg-surface-container-lowest rounded-full overflow-hidden flex ${!isPositive && 'justify-end'}`}>
                          <motion.div 
                            initial={{width: 0}} animate={{width: `${pct}%`}} 
                            transition={{duration: 0.3}}
                            className={`h-full opacity-80 ${isPositive ? 'bg-primary' : 'bg-tertiary'}`} 
                          />
                        </div>
                      </motion.div>
                    )
                  })}
                </AnimatePresence>
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-slate-600 opacity-50">
                   <span className="material-symbols-outlined text-4xl mb-2">biotech</span>
                   <p className="text-[10px] uppercase tracking-widest font-bold">No Elevated Risk</p>
                </div>
              )}
            </div>
          </motion.div>
        </section>

        {/* CENTER STAGE: D3 Map & Chart */}
        <section className="col-span-1 md:col-span-6 flex flex-col gap-4 relative min-w-0 min-h-0">
          <motion.div initial={{opacity:0}} animate={{opacity:1}} transition={{duration:0.6}} className="bg-surface-container-low rounded-2xl border border-outline-variant/10 flex-1 relative overflow-hidden group">
            {/* The actual live NetworkGraph */}
            <div className="absolute inset-0 z-0 opacity-80">
               <NetworkGraph data={networkData} currentData={currentData} />
            </div>
            
            {/* Overlay Map Key */}
            <div className="absolute top-6 left-6 glass p-3 rounded-lg border border-outline-variant/20 z-10 pointer-events-none transition-opacity opacity-0 group-hover:opacity-100">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-2 h-2 rounded-full bg-primary"></div>
                <span className="text-[9px] font-bold uppercase tracking-wider text-white">Nominal Node</span>
              </div>
              <div className="flex items-center gap-2 mb-2">
                <div className="w-2 h-2 rounded-full bg-amber-500"></div>
                <span className="text-[9px] font-bold uppercase tracking-wider text-white">Heat Target</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-error shadow-[0_0_8px_rgba(255,180,171,0.8)]"></div>
                <span className="text-[9px] font-bold uppercase tracking-wider text-white">Critical Anomaly</span>
              </div>
            </div>
          </motion.div>

          {/* Telemetry Flux */}
          <motion.div initial={{opacity:0, y:20}} animate={{opacity:1, y:0}} className="h-48 bg-surface-container rounded-xl border border-outline-variant/10 p-4 relative overflow-hidden flex flex-col shrink-0">
            <div className="flex justify-between items-center mb-2 shrink-0">
              <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-on-surface-variant flex items-center gap-2">
                <span className="material-symbols-outlined text-sm text-secondary-container">water_drop</span>
                Live Telemetry Flux
              </h3>
            </div>
            <div className="flex-1 relative w-full h-full min-h-0">
               <SensorChart
                 history={pressureHistory}
                 nodeNames={networkData?.nodes?.filter(n => n.type === 'Junction').map(n => n.id).slice(0, 5) || []}
               />
            </div>
          </motion.div>
        </section>

        {/* RIGHT COLUMN: Alert Feed */}
        <section className="col-span-1 md:col-span-3 flex flex-col gap-4 overflow-hidden">
          <motion.div initial={{opacity:0, x:20}} animate={{opacity:1, x:0}} className="bg-surface-container rounded-xl p-5 border border-outline-variant/10 flex-1 flex flex-col overflow-hidden">
            <div className="flex justify-between items-center mb-4 shrink-0">
              <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-on-surface-variant flex items-center gap-2">
                 <span className="material-symbols-outlined text-sm text-error">warning</span>
                 Alert Feed
              </h3>
              {alerts.length > 0 && <span className="bg-error/20 border border-error/30 text-error px-2 py-0.5 rounded text-[10px] font-bold mono">{alerts.length} LOGS</span>}
            </div>

            <div className="flex-1 overflow-y-auto space-y-3 pr-2 scroll-smooth">
               {alerts.length === 0 ? (
                 <div className="h-full flex flex-col items-center justify-center text-slate-600 opacity-50">
                    <span className="material-symbols-outlined text-4xl mb-2">done_all</span>
                    <p className="text-[10px] uppercase tracking-widest font-bold">Network Secure</p>
                 </div>
               ) : (
                  <AnimatePresence>
                     {alerts.map((alert, i) => (
                       <motion.div 
                         layout initial={{opacity:0, x:20}} animate={{opacity:1, x:0}}
                         key={alert.id}
                         className={`p-4 rounded-lg border-l-4 relative group transition-colors cursor-default
                           ${i === 0 ? 'bg-surface-container-low border-error hover:bg-surface-container-high' : 'bg-surface-container-lowest border-outline-variant/30 hover:bg-surface-container-low'}
                         `}
                       >
                         <div className="absolute top-0 right-0 p-2 opacity-100">
                           <span className={`mono text-[10px] font-bold ${i===0 ? 'text-error' : 'text-slate-500'}`}>
                             {alert.confidence ? `${(alert.confidence*100).toFixed(0)}% CONF` : ''}
                           </span>
                         </div>
                         <div className="flex items-center gap-2 mb-2">
                           <span className={`material-symbols-outlined text-base ${i===0 ? 'text-error' : 'text-slate-500'}`}>priority_high</span>
                           <span className={`text-[11px] font-bold uppercase tracking-tight ${i===0 ? 'text-white' : 'text-slate-400'}`}>
                              Leak Alert
                           </span>
                         </div>
                         <div className="mono text-[10px] text-primary mb-2 line-clamp-1">
                           TARGET: {alert.suspect_nodes?.join(', ')}
                         </div>
                         <div className="flex justify-between items-center text-[9px] text-slate-500 uppercase font-semibold">
                           <span>{alert.estimated_location || 'GAT Localisation'}</span>
                           <span className="mono">{alert.timestamp || '--'}</span>
                         </div>
                       </motion.div>
                     ))}
                  </AnimatePresence>
               )}
            </div>
          </motion.div>

          {/* AI Insight Mini Panel */}
          <motion.div initial={{opacity:0, y:20}} animate={{opacity:1, y:0}} className="bg-gradient-to-br from-surface-container to-surface-container-highest rounded-xl p-5 border border-outline-variant/10 shrink-0">
            <div className="flex items-center gap-2 mb-3">
              <span className="material-symbols-outlined text-primary text-sm" style={{fontVariationSettings: "'FILL' 1"}}>auto_awesome</span>
              <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-on-surface-variant">System Diagnosis</h3>
            </div>
            <p className="text-xs text-on-surface-variant leading-relaxed font-medium">
               {isAlert ? (
                 <>XGBoost model suggests <span className="text-primary mono font-bold">{lastShapAlert?.suspect_nodes?.join(', ') || 'multiple nodes'}</span> as potential leak locations based on pressure anomaly patterns.</>
               ) : probability > 0.3 ? (
                 <>Elevated XGBoost probability detected. LSTM and GAT models are analysing pressure patterns for confirmation.</>
               ) : (
                 <>All models report normal pressure patterns across the 31-node network. No anomalies detected.</>
               )}
            </p>
          </motion.div>
        </section>

      </main>
    </div>
  )
}

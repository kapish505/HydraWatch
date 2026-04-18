import { ShieldAlert, MapPin, Gauge } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

/**
 * AlertPanel — scrollable list of detected leak alerts.
 * Uses lucide-react icons and framer-motion.
 */
export default function AlertPanel({ alerts }) {
  const severityBadge = (severity) => {
    const s = (severity || 'LOW').toUpperCase()
    const styles = {
      CRITICAL: 'bg-red-500/20 text-red-500 border-red-500/30',
      MEDIUM: 'bg-amber-500/20 text-amber-500 border-amber-500/30',
      LOW: 'bg-emerald-500/20 text-emerald-500 border-emerald-500/30',
    }
    return (
      <span className={`text-[9px] font-bold uppercase px-2 py-0.5 rounded-full border ${styles[s] || styles.LOW}`}>
        {s}
      </span>
    )
  }

  return (
    <div className="glass-panel p-5 h-full flex flex-col border border-white/10 relative overflow-hidden">
      <div className="flex items-center justify-between mb-4 relative z-10">
        <h2 className="text-xs font-semibold text-white/80 uppercase tracking-widest flex items-center gap-2">
          <ShieldAlert size={14} className="text-red-400" />
          Alert Feed
        </h2>
        <span className="text-[10px] text-red-400 bg-red-500/10 border border-red-500/20 px-2 py-0.5 rounded-md font-mono font-bold">
          {alerts.length}
        </span>
      </div>

      <div className="flex-1 overflow-y-auto space-y-3 pr-2 relative z-10 hide-scrollbar">
        {alerts.length === 0 ? (
          <motion.div 
            initial={{ opacity: 0 }} 
            animate={{ opacity: 1 }} 
            className="h-full flex flex-col items-center justify-center text-center text-slate-500"
          >
            <ShieldAlert size={32} className="mb-3 opacity-20" />
            <div className="text-sm font-medium">No alerts detected</div>
            <div className="text-[10px] mt-1 opacity-60">System running nominally</div>
          </motion.div>
        ) : (
          <AnimatePresence mode="popLayout">
            {alerts.map((alert, i) => (
              <motion.div
                layout
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.3 }}
                key={alert.id || i}
                className={`p-3.5 rounded-xl border transition-all ${
                  i === 0 && alert.severity === 'CRITICAL'
                    ? 'bg-gradient-to-br from-red-950/40 to-black/40 border-red-500/30 shadow-[0_0_15px_rgba(239,68,68,0.1)]'
                    : 'bg-black/40 border-white/5 hover:bg-white/5'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  {severityBadge(alert.severity)}
                  <span className="text-[10px] text-slate-400 font-mono flex items-center gap-1 bg-white/5 px-1.5 py-0.5 rounded">
                    <Gauge size={10} />
                    {alert.confidence ? `${(alert.confidence * 100).toFixed(0)}%` : ''}
                  </span>
                </div>

                <div className="text-xs text-white font-medium mb-1.5 flex items-start gap-1.5">
                  <span className="text-red-400 mt-0.5">•</span>
                  <span>
                    {alert.suspect_nodes?.length > 0
                      ? `Suspect: ${alert.suspect_nodes.join(', ')}`
                      : 'Leak trajectory detected'}
                  </span>
                </div>

                {alert.estimated_location && (
                  <div className="text-[10px] text-slate-400 mb-2 flex items-center gap-1.5">
                    <MapPin size={10} className="text-amber-400" />
                    <span>{alert.estimated_location}</span>
                  </div>
                )}

                <div className="text-[10px] text-slate-500 font-mono tracking-wider">
                  {alert.detected_at || alert.timestamp || '--'}
                </div>

                {/* SHAP top 3 features */}
                {alert.shap_features?.length > 0 && (
                  <div className="mt-3 pt-2.5 border-t border-white/5 space-y-1.5">
                    {alert.shap_features.slice(0, 3).map((feat, j) => (
                      <div key={j} className="flex items-center justify-between text-[9px]">
                        <span className="text-slate-400 truncate mr-2 font-mono uppercase">
                          {feat.feature?.replace('Node_', 'N').replace('_', ' ')}
                        </span>
                        <span className="text-red-400 font-mono shrink-0 bg-red-500/10 px-1 rounded">
                          {typeof feat.impact === 'number' ? `+${feat.impact.toFixed(3)}` : ''}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>
        )}
      </div>
    </div>
  )
}

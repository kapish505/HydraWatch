import { motion } from 'framer-motion'
import { FileSearch, Sparkles } from 'lucide-react'

/**
 * ShapWaterfall — horizontal bar chart showing SHAP feature importance.
 * Uses lucide-react.
 */
export default function ShapWaterfall({ alert }) {
  const features = alert?.shap_features || []

  // Sort by abs impact descending, take top 8
  const sorted = [...features]
    .sort((a, b) => Math.abs(b.impact || 0) - Math.abs(a.impact || 0))
    .slice(0, 8)

  const maxImpact = sorted.length > 0
    ? Math.max(...sorted.map(f => Math.abs(f.impact || 0)), 0.01)
    : 1

  const formatFeature = (name = '') =>
    name.replace('Node_', 'N').replace(/_/g, ' ').replace('pressure ', 'P:').replace('rolling ', 'R:').slice(0, 18)

  return (
    <div className="glass-panel p-5 flex flex-col gap-3 flex-1 min-h-0 border border-white/10 group">
      <div className="flex items-center justify-between shrink-0">
        <h2 className="text-xs font-semibold text-white/80 uppercase tracking-widest flex items-center gap-2">
          <FileSearch size={14} className="text-cyan-400" />
          TreeSHAP Matrix
        </h2>
        <Sparkles size={12} className="text-slate-500 opacity-50 group-hover:opacity-100 transition-opacity" />
      </div>

      {sorted.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center text-center text-slate-500">
          <FileSearch size={28} className="mb-2 opacity-20" />
          <div className="text-xs font-medium">Model features pending</div>
          <div className="text-[10px] mt-0.5 opacity-60">Waiting for anomaly trigger</div>
        </div>
      ) : (
        <div className="flex-1 flex flex-col justify-end space-y-2 pr-1 pb-1">
          {sorted.map((feat, i) => {
            const pct = Math.min(Math.abs(feat.impact || 0) / maxImpact * 100, 100)
            const isPositive = (feat.impact || 0) > 0

            return (
              <div key={i} className="flex flex-col gap-1 w-full">
                <div className="flex items-center justify-between text-[9px] uppercase tracking-wider">
                  <span className="text-slate-400 font-mono truncate mr-2">
                    {formatFeature(feat.feature)}
                  </span>
                  <span className={`font-mono shrink-0 font-bold ${isPositive ? 'text-red-400' : 'text-emerald-400'}`}>
                    {isPositive ? '+' : ''}{(feat.impact || 0).toFixed(3)}
                  </span>
                </div>
                <div className="w-full bg-black/40 border border-white/5 rounded-full h-1.5 overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${pct}%` }}
                    transition={{ duration: 0.5, delay: i * 0.05 }}
                    className={`h-full rounded-full ${isPositive ? 'bg-gradient-to-r from-red-600 to-red-400' : 'bg-gradient-to-r from-emerald-600 to-emerald-400'}`}
                  />
                </div>
              </div>
            )
          })}

          <div className="flex justify-between text-[9px] text-slate-500 pt-2 mt-2 border-t border-white/10 font-bold uppercase tracking-widest">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 bg-emerald-500 rounded-sm inline-block" /> Drives Nominal
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 bg-red-500 rounded-sm inline-block" /> Drives Leak
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

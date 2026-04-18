import React from 'react';
import { Link } from 'react-router-dom';
import { motion, useScroll, useTransform } from 'framer-motion';

const FadeIn = ({ children, delay = 0, className = "" }) => (
  <motion.div
    initial={{ opacity: 0, y: 30 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true, margin: "-50px" }}
    transition={{ duration: 0.7, delay, ease: [0.21, 0.47, 0.32, 0.98] }}
    className={className}
  >
    {children}
  </motion.div>
);

export default function Home() {
  const { scrollYProgress } = useScroll();
  const y = useTransform(scrollYProgress, [0, 1], ["0%", "40%"]);

  return (
    <div className="min-h-screen bg-background text-on-surface selection:bg-primary/30 overflow-x-hidden">
      
      {/* Top Navigation */}
      <nav className="fixed top-0 w-full z-50 bg-[#11131b]/60 backdrop-blur-xl border-b border-[#3d494c]/20 shadow-[0_0_20px_rgba(6,182,212,0.1)] h-16 flex justify-between items-center px-8">
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="text-xl font-bold tracking-tighter text-primary uppercase font-headline"
        >
          HydraWatch
        </motion.div>
        <motion.div 
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="hidden md:flex gap-8 items-center"
        >
          <a className="text-primary border-b-2 border-primary pb-1 font-medium tracking-tight" href="#">Models</a>
          <a className="text-slate-400 font-medium hover:text-primary transition-colors duration-300 tracking-tight" href="#">Simulation</a>
          <a className="text-slate-400 font-medium hover:text-primary transition-colors duration-300 tracking-tight" href="#">Telemetry</a>
          <a className="text-slate-400 font-medium hover:text-primary transition-colors duration-300 tracking-tight" href="#">Network</a>
        </motion.div>
        <motion.div 
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="flex items-center gap-6"
        >
          <div className="flex gap-4 text-slate-400">
            <span className="material-symbols-outlined cursor-pointer hover:text-primary transition-colors">settings</span>
            <span className="material-symbols-outlined cursor-pointer hover:text-primary transition-colors">notifications_active</span>
          </div>
          <Link to="/dashboard" className="bg-gradient-to-br from-primary to-primary-container text-on-primary font-bold px-6 py-2 rounded-lg active:scale-95 transition-transform hover:shadow-lg hover:shadow-primary/20">
            Launch Console
          </Link>
        </motion.div>
      </nav>

      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center pt-16 overflow-hidden">
        {/* Background Ambient Glows */}
        <motion.div style={{ y }} className="absolute inset-0 pointer-events-none">
           <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-[120px]"></div>
           <div className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] bg-secondary-container/10 rounded-full blur-[150px]"></div>
        </motion.div>

        <div className="container mx-auto px-8 relative z-10 text-center">
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass border border-outline-variant/20 mb-8"
          >
            <span className="w-2 h-2 rounded-full bg-tertiary status-pulse"></span>
            <span className="text-xs font-mono uppercase tracking-[0.2em] text-on-surface-variant font-medium">System Status: Live Monitoring</span>
          </motion.div>

          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.1 }}
            className="text-6xl md:text-8xl font-extrabold tracking-tighter mb-6 bg-clip-text text-transparent bg-gradient-to-b from-white to-white/50 leading-none font-headline"
          >
            HydraWatch: Water <br/> Network Intelligence
          </motion.h1>

          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="max-w-2xl mx-auto text-on-surface-variant text-lg mb-12 font-light"
          >
            Ensemble leak detection for water distribution networks. Combining XGBoost, LSTM Autoencoder, and Graph Attention Networks to detect and localise pipe leaks from pressure sensor data.
          </motion.p>
          
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="flex flex-col md:flex-row items-center justify-center gap-6"
          >
            <Link to="/dashboard" className="px-8 py-4 bg-gradient-to-br from-primary to-primary-container text-on-primary-container font-bold rounded-lg shadow-lg hover:shadow-primary/20 transition-all active:scale-95">
                Launch Simulation
            </Link>
            <button className="px-8 py-4 glass border border-outline-variant/30 text-primary font-bold rounded-lg hover:bg-white/5 transition-all active:scale-95">
                LeakDB Dataset
            </button>
          </motion.div>
        </div>

        {/* Decorative UI Element */}
        <div className="absolute bottom-0 w-full h-64 bg-gradient-to-t from-background to-transparent pointer-events-none"></div>
      </section>

      {/* The Problem Section */}
      <section className="py-24 relative">
        <div className="container mx-auto px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            <FadeIn className="space-y-8">
              <h2 className="text-4xl font-bold tracking-tight text-white font-headline">The Evolution of Leak Detection</h2>
              <p className="text-on-surface-variant leading-relaxed">
                  Traditional Minimum Night Flow (MNF) methods rely on daily overnight flow reviews, achieving F1 ≈ 0.50 with detection delays of up to 6 hours. HydraWatch uses continuous pressure monitoring with ML for faster, more accurate detection.
              </p>
              <div className="space-y-4">
                <div className="flex items-start gap-4 p-6 rounded-xl border border-outline-variant/10 bg-surface-container-lowest">
                  <span className="material-symbols-outlined text-error mt-1" style={{fontVariationSettings: "'FILL' 1"}}>hearing</span>
                  <div>
                    <h4 className="font-bold text-white">Legacy MNF Detection</h4>
                    <p className="text-sm text-on-surface-variant">Daily overnight flow review. F1 ≈ 0.50, detection delay ~6 hours.</p>
                  </div>
                </div>
                <div className="flex items-start gap-4 p-6 rounded-xl border border-primary/20 bg-surface-container-low shadow-inner">
                  <span className="material-symbols-outlined text-primary mt-1" style={{fontVariationSettings: "'FILL' 1"}}>hub</span>
                  <div>
                    <h4 className="font-bold text-white">HydraWatch Ensemble</h4>
                    <p className="text-sm text-on-surface-variant">3-model ensemble on 31-node Hanoi network. XGBoost F1 = 0.56, LSTM Recall = 84.7%.</p>
                  </div>
                </div>
              </div>
            </FadeIn>
            
            <FadeIn delay={0.2} className="relative aspect-square">
              <div className="absolute inset-0 glass rounded-full border border-outline-variant/20 overflow-hidden">
                <img className="w-full h-full object-cover opacity-40 mix-blend-screen" alt="Digital water blueprint" src="https://lh3.googleusercontent.com/aida-public/AB6AXuCZLWxH5lOmq52nn5tx2ojsWakgkufhzgKXQWGQMiIrRBxtJo1uNmT5bx2ckILUt9uKe-koUbKWPN0aKr606zqTJn_6RlDNqvdQ2xgVio2BUBOwKTq4sC8tbGKBcJELcxPLPHElLrI00Z9MjwEBZTBTFNa7psLsd6szVkr-R7F1-wgK3ML_wC1Yn_4wvmnKsS69beISPaKu1hgT88AJ0qWkjYOuDGsN4KPqBfFWNE848x1pk_qwZj3hPMdNRvfpjqd3_ylaJ2YXRDSV"/>
              </div>
              {/* Telemetry Floating Cards */}
              <motion.div 
                animate={{ y: [-5, 5, -5] }}
                transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
                className="absolute top-10 -left-10 glass p-4 rounded-lg border border-primary/30 glow-cyan w-48"
              >
                <div className="flex justify-between items-center mb-2">
                  <span className="text-[10px] font-mono text-primary uppercase font-bold">Pressure</span>
                  <span className="w-2 h-2 rounded-full bg-tertiary"></span>
                </div>
                <div className="text-2xl font-mono text-white">38.2 <span className="text-xs text-slate-500">m</span></div>
              </motion.div>

              <motion.div 
                animate={{ y: [5, -5, 5] }}
                transition={{ duration: 7, repeat: Infinity, ease: "easeInOut", delay: 1 }}
                className="absolute bottom-20 -right-10 glass p-4 rounded-lg border border-secondary-container/30 w-56"
              >
                <div className="flex justify-between items-center mb-2">
                  <span className="text-[10px] font-mono text-secondary uppercase font-bold">XGBoost Score</span>
                </div>
                <div className="text-2xl font-mono text-white">0.56 <span className="text-xs text-slate-500">F1</span></div>
                <div className="mt-2 h-1 w-full bg-surface-container rounded-full overflow-hidden">
                  <div className="h-full bg-secondary-container w-3/4"></div>
                </div>
              </motion.div>
            </FadeIn>
          </div>
        </div>
      </section>

      {/* 3-Model Approach */}
      <section className="py-24 bg-surface-container-lowest/50">
        <div className="container mx-auto px-8">
          <FadeIn className="mb-16 text-center">
            <h3 className="text-3xl font-bold mb-4 font-headline text-white">The Neural Architecture</h3>
            <p className="text-on-surface-variant">A multi-modal ensemble approach to infrastructure health monitoring.</p>
          </FadeIn>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <FadeIn delay={0.1}>
              <div className="group p-8 rounded-2xl glass border border-outline-variant/10 hover:border-primary/50 transition-all duration-500 relative overflow-hidden h-full">
                <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                  <span className="material-symbols-outlined text-6xl" style={{fontVariationSettings: "'FILL' 1"}}>timeline</span>
                </div>
                <div className="mb-6 inline-flex items-center justify-center w-12 h-12 rounded-lg bg-primary/10 text-primary">
                  <span className="material-symbols-outlined">auto_graph</span>
                </div>
                <h4 className="text-xl font-bold mb-3 text-white font-headline">LSTM Autoencoder</h4>
                <p className="text-sm text-on-surface-variant mb-6 leading-relaxed">Learns normal pressure patterns from 400 training scenarios. Flags deviations via reconstruction error against an F1-optimised threshold. Recall: 84.7%.</p>
                <div className="pt-4 border-t border-outline-variant/10 mt-auto">
                  <span className="text-[10px] font-mono text-primary uppercase tracking-widest font-bold">Focus: Sequence Patterns</span>
                </div>
              </div>
            </FadeIn>

            <FadeIn delay={0.2}>
              <div className="group p-8 rounded-2xl bg-surface-container border border-outline-variant/10 hover:border-secondary-container/50 transition-all duration-500 relative overflow-hidden h-full">
                <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                  <span className="material-symbols-outlined text-6xl" style={{fontVariationSettings: "'FILL' 1"}}>hub</span>
                </div>
                <div className="mb-6 inline-flex items-center justify-center w-12 h-12 rounded-lg bg-secondary-container/10 text-secondary">
                  <span className="material-symbols-outlined">account_tree</span>
                </div>
                <h4 className="text-xl font-bold mb-3 text-white font-headline">GAT (Graph Attention)</h4>
                <p className="text-sm text-on-surface-variant mb-6 leading-relaxed">3-layer graph neural network over the 31-node Hanoi topology. Narrows leak search to top-3 suspect nodes — 2.7x better than random inspection.</p>
                <div className="pt-4 border-t border-outline-variant/10 mt-auto">
                  <span className="text-[10px] font-mono text-secondary uppercase tracking-widest font-bold">Focus: Network Topology</span>
                </div>
              </div>
            </FadeIn>

            <FadeIn delay={0.3}>
              <div className="group p-8 rounded-2xl glass border border-outline-variant/10 hover:border-tertiary/50 transition-all duration-500 relative overflow-hidden h-full">
                <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                  <span className="material-symbols-outlined text-6xl" style={{fontVariationSettings: "'FILL' 1"}}>psychology</span>
                </div>
                <div className="mb-6 inline-flex items-center justify-center w-12 h-12 rounded-lg bg-tertiary/10 text-tertiary">
                  <span className="material-symbols-outlined">analytics</span>
                </div>
                <h4 className="text-xl font-bold mb-3 text-white font-headline">XGBoost &amp; TreeSHAP</h4>
                <p className="text-sm text-on-surface-variant mb-6 leading-relaxed">Gradient-boosted classifier on 13 engineered features per node. TreeSHAP provides per-feature impact scores so operators see exactly which sensor triggered each alert. Val F1: 0.56.</p>
                <div className="pt-4 border-t border-outline-variant/10 mt-auto">
                  <span className="text-[10px] font-mono text-tertiary uppercase tracking-widest font-bold">Focus: Confidence &amp; Logic</span>
                </div>
              </div>
            </FadeIn>
          </div>
        </div>
      </section>

      {/* Dataset Credits Section */}
      <section className="py-24">
        <div className="container mx-auto px-8">
          <FadeIn className="p-12 rounded-3xl bg-gradient-to-br from-surface-container-low to-surface-container border border-outline-variant/10 flex flex-col md:flex-row items-center justify-between gap-12">
            <div className="md:max-w-xl">
              <h3 className="text-2xl font-bold mb-4 font-headline text-white">Trained on LeakDB Benchmarks</h3>
              <p className="text-on-surface-variant">Models trained and validated on 500 leak scenarios from the KIOS LeakDB Hanoi network — an open-source hydraulic benchmark with realistic demand patterns and leak signatures.</p>
            </div>
            <div className="flex gap-8 items-center grayscale opacity-60 hover:grayscale-0 hover:opacity-100 transition-all">
              <div className="text-center">
                <div className="text-2xl font-bold text-white mb-1 font-headline tracking-tight">LeakDB Hanoi</div>
                <div className="text-[10px] font-mono text-primary uppercase font-bold">500 Scenarios</div>
              </div>
              <div className="h-12 w-px bg-outline-variant/30"></div>
              <div className="text-center">
                <div className="text-2xl font-bold text-white mb-1 font-headline tracking-tight">31 Nodes</div>
                <div className="text-[10px] font-mono text-primary uppercase font-bold">34 Pipes</div>
              </div>
            </div>
          </FadeIn>
        </div>
      </section>

      {/* Footer */}
      <footer className="w-full py-12 border-t border-outline-variant/10 mt-12 bg-surface-container-lowest">
        <div className="max-w-7xl mx-auto px-8 flex flex-col items-center gap-6">
          <div className="font-bold text-primary text-xl font-headline tracking-tighter uppercase">HydraWatch <span className="opacity-50">Intelligence</span></div>
          <div className="flex flex-wrap justify-center gap-8 text-slate-500 text-sm">
            <a className="hover:text-white transition-colors" href="https://github.com/KIOS-Research/LeakDB" target="_blank" rel="noreferrer">LeakDB Dataset</a>
            <a className="hover:text-white transition-colors" href="/dashboard">Dashboard</a>
          </div>
          <div className="text-slate-500 text-xs text-center max-w-lg mt-4">
            © <span className="font-mono">2026</span> HydraWatch — Water Network Intelligence System. Built with PyTorch, XGBoost, and FastAPI.
          </div>
        </div>
      </footer>
    </div>
  );
}

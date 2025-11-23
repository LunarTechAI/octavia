"use client";

import { motion } from "framer-motion";
import { Upload, Video, Sparkles, CheckCircle, Rocket } from "lucide-react";

export default function VideoTranslationPage() {
    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex flex-col gap-2">
                <h1 className="font-display text-3xl font-black text-white text-glow-purple">Video AI Translator</h1>
                <p className="text-slate-400 text-sm">Upload your video and translate it across languages</p>
            </div>

            {/* Upload Zone */}
            <motion.div
                whileHover={{ scale: 1.01 }}
                className="glass-panel glass-panel-high relative border-2 border-dashed border-primary-purple/30 hover:border-primary-purple/50 transition-all cursor-pointer group mb-6 overflow-hidden"
            >
                <div className="glass-shine" />
                <div className="glow-purple" style={{ width: "300px", height: "300px", top: "50%", left: "50%", transform: "translate(-50%, -50%)", zIndex: 1 }} />

                <div className="relative z-20 py-12 px-6">
                    <div className="flex flex-col items-center justify-center gap-3 text-center">
                        <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-primary-purple/10 border border-primary-purple/30 shadow-glow group-hover:scale-110 transition-transform">
                            <Video className="w-8 h-8 text-primary-purple-bright" />
                        </div>
                        <div>
                            <h3 className="text-white text-lg font-bold mb-1 text-glow-purple">Drop your video here</h3>
                            <p className="text-slate-400 text-sm">or click to browse files • MP4, AVI, MOV supported</p>
                        </div>
                    </div>
                </div>
            </motion.div>

            {/* Configuration */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                {/* Source Language */}
                <div className="glass-card p-4">
                    <label className="text-white text-sm font-semibold mb-2 block">Source Language</label>
                    <select className="glass-select w-full">
                        <option>English</option>
                        <option>Spanish</option>
                        <option>French</option>
                        <option>German</option>
                        <option>Italian</option>
                    </select>
                </div>

                {/* Target Language */}
                <div className="glass-card p-4">
                    <label className="text-white text-sm font-semibold mb-2 block">Target Language</label>
                    <select className="glass-select w-full">
                        <option>Spanish</option>
                        <option>English</option>
                        <option>French</option>
                        <option>German</option>
                        <option>Italian</option>
                    </select>
                </div>
            </div>

            {/* AI Options */}
            <div className="glass-panel glass-panel-glow mb-6 p-5 relative overflow-hidden">
                <div className="glass-shine" />
                <div className="relative z-10">
                    <div className="flex items-start gap-3 mb-3">
                        <Sparkles className="w-5 h-5 text-accent-cyan" />
                        <div>
                            <h3 className="text-white text-sm font-bold mb-1">AI-Powered Features</h3>
                            <p className="text-slate-400 text-xs">Enhance your translation with advanced AI capabilities</p>
                        </div>
                    </div>
                    <div className="flex flex-wrap gap-2">
                        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-accent-cyan/10 border border-accent-cyan/30">
                            <CheckCircle className="w-3.5 h-3.5 text-accent-cyan" />
                            <span className="text-slate-200 text-xs font-medium">Voice Synthesis</span>
                        </div>
                        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-primary-purple/10 border border-primary-purple/30">
                            <CheckCircle className="w-3.5 h-3.5 text-primary-purple-bright" />
                            <span className="text-slate-200 text-xs font-medium">Lip Sync</span>
                        </div>
                        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-accent-pink/10 border border-accent-pink/30">
                            <CheckCircle className="w-3.5 h-3.5 text-accent-pink" />
                            <span className="text-slate-200 text-xs font-medium">Subtitle Generation</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Start Button */}
            <button className="btn-border-beam w-full group">
                <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-4 text-base">
                    <Rocket className="w-5 h-5 group-hover:rotate-12 transition-transform duration-300" />
                    <span>Start Translation</span>
                </div>
            </button>
        </div>
    );
}

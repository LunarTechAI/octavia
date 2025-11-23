"use client";

import { motion } from "framer-motion";
import { FileText, Languages } from "lucide-react";

export default function SubtitleTranslatePage() {
    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex flex-col gap-2">
                <h1 className="font-display text-3xl font-black text-white text-glow-purple">Subtitle Translation</h1>
                <p className="text-slate-400 text-sm">Translate existing subtitle files to another language</p>
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
                            <FileText className="w-8 h-8 text-primary-purple-bright" />
                        </div>
                        <div>
                            <h3 className="text-white text-lg font-bold mb-1 text-glow-purple">Drop subtitle file here</h3>
                            <p className="text-slate-400 text-sm">or click to browse files • SRT, VTT, ASS supported</p>
                        </div>
                    </div>
                </div>
            </motion.div>

            {/* Configuration */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                <div className="glass-card p-4">
                    <label className="text-white text-sm font-semibold mb-2 block">Source Language</label>
                    <select className="glass-select w-full">
                        <option>English</option>
                        <option>Spanish</option>
                        <option>French</option>
                        <option>German</option>
                    </select>
                </div>
                <div className="glass-card p-4">
                    <label className="text-white text-sm font-semibold mb-2 block">Target Language</label>
                    <select className="glass-select w-full">
                        <option>Spanish</option>
                        <option>English</option>
                        <option>French</option>
                        <option>German</option>
                    </select>
                </div>
            </div>

            {/* Start Button */}
            <button className="btn-border-beam w-full group">
                <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-4 text-base">
                    <Languages className="w-5 h-5" />
                    <span>Translate Subtitles</span>
                </div>
            </button>
        </div>
    );
}

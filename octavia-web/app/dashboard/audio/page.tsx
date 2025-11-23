"use client";

import { motion } from "framer-motion";
import { AudioLines, Play } from "lucide-react";

export default function AudioTranslationPage() {
    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex flex-col gap-2">
                <h1 className="font-display text-3xl font-black text-white text-glow-purple">Audio Translation</h1>
                <p className="text-slate-400 text-sm">Transform audio files across languages with AI-powered voice synthesis</p>
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
                            <AudioLines className="w-8 h-8 text-primary-purple-bright" />
                        </div>
                        <div>
                            <h3 className="text-white text-lg font-bold mb-1 text-glow-purple">Drop your audio here</h3>
                            <p className="text-slate-400 text-sm">or click to browse files • MP3, WAV, FLAC supported</p>
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
                    </select>
                </div>
            </div>

            {/* Voice Options */}
            <div className="glass-panel glass-panel-glow mb-6 p-5 relative overflow-hidden">
                <div className="glass-shine" />
                <div className="relative z-10">
                    <div className="flex items-start gap-3 mb-3">
                        <AudioLines className="w-5 h-5 text-primary-purple-bright" />
                        <div>
                            <h3 className="text-white text-sm font-bold mb-1">Voice Synthesis Options</h3>
                            <p className="text-slate-400 text-xs">Choose voice characteristics for translated audio</p>
                        </div>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                        <div className="glass-card p-3">
                            <label className="text-white text-xs font-semibold mb-2 block">Voice Gender</label>
                            <select className="glass-select w-full text-sm">
                                <option>Male</option>
                                <option>Female</option>
                                <option>Neutral</option>
                            </select>
                        </div>
                        <div className="glass-card p-3">
                            <label className="text-white text-xs font-semibold mb-2 block">Voice Style</label>
                            <select className="glass-select w-full text-sm">
                                <option>Natural</option>
                                <option>Professional</option>
                                <option>Casual</option>
                            </select>
                        </div>
                    </div>
                </div>
            </div>

            {/* Start Button */}
            <button className="btn-border-beam w-full group">
                <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-4 text-base">
                    <Play className="w-5 h-5 group-hover:rotate-12 transition-transform duration-300" />
                    <span>Start Audio Translation</span>
                </div>
            </button>
        </div>
    );
}

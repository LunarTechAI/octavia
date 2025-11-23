"use client";

import { motion } from "framer-motion";
import { Download, CheckCircle, Edit3 } from "lucide-react";

export default function SubtitleReviewPage() {
    const subtitles = [
        { id: 1, timestamp: "00:00:00 → 00:00:05", text: "Welcome to this amazing video tutorial." },
        { id: 2, timestamp: "00:00:06 → 00:00:11", text: "Today we'll explore advanced techniques for subtitle generation." },
        { id: 3, timestamp: "00:00:12 → 00:00:17", text: "The AI has analyzed the audio and created these timestamps." },
        { id: 4, timestamp: "00:00:18 → 00:00:23", text: "You can edit any subtitle by clicking on it." },
        { id: 5, timestamp: "00:00:24 → 00:00:30", text: "When you're satisfied, download the final file." },
    ];

    return (
        <div className="space-y-8">
            <div className="flex flex-col lg:flex-row gap-8">
                {/* Left: Subtitle Editor */}
                <div className="flex-1 flex flex-col gap-6">
                    <div>
                        <h1 className="font-display text-3xl font-black text-white text-glow-purple mb-2">Review Subtitles</h1>
                        <p className="text-slate-400 text-sm">Edit and refine your generated subtitles</p>
                    </div>

                    {/* Subtitle List */}
                    <div className="glass-panel p-6 space-y-3 max-h-[600px] overflow-y-auto custom-scrollbar">
                        {subtitles.map((sub) => (
                            <motion.div
                                key={sub.id}
                                whileHover={{ scale: 1.01 }}
                                className="glass-card p-4 hover:border-primary-purple/30 transition-all cursor-pointer group"
                            >
                                <div className="flex items-start justify-between gap-3 mb-2">
                                    <span className="text-xs text-slate-500 font-mono">{sub.timestamp}</span>
                                    <button className="opacity-0 group-hover:opacity-100 transition-opacity">
                                        <Edit3 className="w-3.5 h-3.5 text-slate-400 hover:text-white" />
                                    </button>
                                </div>
                                <p className="text-sm text-white leading-relaxed">{sub.text}</p>
                            </motion.div>
                        ))}
                    </div>
                </div>

                {/* Right: Actions & Stats */}
                <div className="w-full lg:w-96 flex flex-col gap-6">
                    <div className="glass-panel p-6">
                        <h3 className="text-lg font-bold text-white mb-4">Subtitle Stats</h3>
                        <div className="space-y-3">
                            <div className="flex items-center gap-4">
                                <div className="flex size-10 items-center justify-center rounded-full bg-green-500/20 text-green-400 shadow-glow">
                                    <CheckCircle className="w-5 h-5" />
                                </div>
                                <div>
                                    <p className="text-slate-400 text-xs font-medium uppercase tracking-wider">Total Lines</p>
                                    <p className="text-white text-lg font-bold">147</p>
                                </div>
                            </div>
                            <div className="flex items-center gap-4">
                                <div className="flex size-10 items-center justify-center rounded-full bg-blue-500/20 text-blue-400 shadow-glow">
                                    <CheckCircle className="w-5 h-5" />
                                </div>
                                <div>
                                    <p className="text-slate-400 text-xs font-medium uppercase tracking-wider">Duration</p>
                                    <p className="text-white text-lg font-bold">12:34</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Actions */}
                    <div className="flex flex-col gap-4">
                        <button className="btn-border-beam w-full group">
                            <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-3">
                                <Download className="w-5 h-5 group-hover:translate-y-1 transition-transform" />
                                <span>Download SRT</span>
                            </div>
                        </button>

                        <div className="grid grid-cols-2 gap-3">
                            <button className="flex items-center justify-center gap-2 py-2.5 rounded-lg border border-white/10 bg-white/5 text-white font-medium hover:bg-white/10 transition-all text-sm">
                                Download VTT
                            </button>
                            <button className="flex items-center justify-center gap-2 py-2.5 rounded-lg border border-white/10 bg-white/5 text-white font-medium hover:bg-white/10 transition-all text-sm">
                                Download ASS
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

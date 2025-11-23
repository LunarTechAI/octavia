"use client";

import { motion } from "framer-motion";
import { Plus, Mic, Play, MoreVertical, Trash2, Edit2 } from "lucide-react";

const voices = [
    { id: 1, name: "Narrator - Deep", type: "Cloned", language: "English (US)", date: "2 days ago" },
    { id: 2, name: "Sarah - Professional", type: "Cloned", language: "English (UK)", date: "1 week ago" },
    { id: 3, name: "Promo - Energetic", type: "Synthetic", language: "Spanish", date: "2 weeks ago" },
];

export default function MyVoicesPage() {
    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="font-display text-3xl font-black text-white mb-2 text-glow-purple">My Voices</h1>
                    <p className="text-slate-400 text-sm">Manage your cloned and synthetic voice library</p>
                </div>
                <button className="btn-border-beam">
                    <div className="btn-border-beam-inner flex items-center gap-2 px-6 py-2.5">
                        <Plus className="w-4 h-4" />
                        <span>Add New Voice</span>
                    </div>
                </button>
            </div>

            {/* Voices Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
                {/* Add New Card (Alternative) */}
                <motion.div
                    whileHover={{ scale: 1.02 }}
                    className="glass-panel border-dashed border-white/10 flex flex-col items-center justify-center p-8 cursor-pointer hover:bg-white/5 hover:border-primary-purple/30 group transition-all min-h-[200px]"
                >
                    <div className="w-12 h-12 rounded-full bg-white/5 flex items-center justify-center mb-4 group-hover:bg-primary-purple/20 transition-colors">
                        <Plus className="w-6 h-6 text-slate-400 group-hover:text-primary-purple-bright" />
                    </div>
                    <h3 className="text-white font-bold mb-1">Create Voice</h3>
                    <p className="text-slate-500 text-xs">Clone a new voice or design one</p>
                </motion.div>

                {voices.map((voice) => (
                    <motion.div
                        key={voice.id}
                        whileHover={{ y: -4 }}
                        className="glass-panel-glow p-5 relative group"
                    >
                        <div className="glass-shine" />

                        <div className="flex justify-between items-start mb-4 relative z-10">
                            <div className="w-10 h-10 rounded-lg bg-primary-purple/10 flex items-center justify-center border border-primary-purple/20">
                                <Mic className="w-5 h-5 text-primary-purple-bright" />
                            </div>
                            <button className="text-slate-500 hover:text-white transition-colors">
                                <MoreVertical className="w-4 h-4" />
                            </button>
                        </div>

                        <div className="relative z-10">
                            <h3 className="text-white font-bold text-lg mb-1">{voice.name}</h3>
                            <div className="flex items-center gap-2 mb-4">
                                <span className="text-xs px-2 py-0.5 rounded-full bg-white/5 text-slate-300 border border-white/10">
                                    {voice.type}
                                </span>
                                <span className="text-xs text-slate-500">•</span>
                                <span className="text-xs text-slate-500">{voice.language}</span>
                            </div>

                            <div className="flex items-center gap-2 mt-4 pt-4 border-t border-white/5">
                                <button className="flex-1 flex items-center justify-center gap-2 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-xs font-medium text-white transition-colors">
                                    <Play className="w-3 h-3 fill-current" />
                                    Preview
                                </button>
                                <button className="p-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition-colors">
                                    <Edit2 className="w-3.5 h-3.5" />
                                </button>
                                <button className="p-2 rounded-lg hover:bg-red-500/10 text-slate-400 hover:text-red-400 transition-colors">
                                    <Trash2 className="w-3.5 h-3.5" />
                                </button>
                            </div>
                        </div>
                    </motion.div>
                ))}
            </div>
        </div>
    );
}

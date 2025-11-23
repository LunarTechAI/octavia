"use client";

import { motion } from "framer-motion";
import { CheckCircle2, Loader2, Clock } from "lucide-react";

export default function SubtitleGenerationProgressPage() {
    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex flex-col gap-1 border-b border-white/10 pb-6">
                <h1 className="font-display text-3xl font-black text-white text-glow-purple">Generating Subtitles...</h1>
                <p className="text-slate-400 text-sm">Processing your media file and extracting speech</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="flex flex-col gap-6">
                    {/* Overall Progress */}
                    <div className="glass-panel p-6">
                        <div className="flex items-center justify-between gap-6 mb-3">
                            <p className="text-base font-medium text-white">Overall Progress</p>
                            <p className="text-2xl font-bold text-primary-purple-bright text-glow-purple">65%</p>
                        </div>
                        <div className="w-full bg-white/5 rounded-full h-2.5 mb-3 overflow-hidden">
                            <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: "65%" }}
                                transition={{ duration: 1, ease: "easeOut" }}
                                className="bg-primary-purple h-2.5 rounded-full shadow-glow"
                            />
                        </div>
                        <p className="text-sm text-slate-400">Transcribing audio segment 12 of 18...</p>
                    </div>

                    {/* Pipeline Steps */}
                    <div className="glass-panel p-6">
                        <h2 className="text-lg font-bold text-white mb-4">Processing Steps</h2>
                        <div className="space-y-4">
                            {/* Step 1: Complete */}
                            <div className="flex items-center gap-4">
                                <div className="flex size-10 items-center justify-center rounded-full bg-green-500/20 text-green-400 shadow-glow">
                                    <CheckCircle2 className="w-5 h-5" />
                                </div>
                                <div className="flex-1">
                                    <p className="text-sm font-medium text-white">Audio Extraction</p>
                                    <p className="text-xs text-slate-400">Completed</p>
                                </div>
                            </div>

                            {/* Step 2: In Progress */}
                            <div className="flex items-center gap-4">
                                <div className="flex size-10 items-center justify-center rounded-full bg-primary-purple/20 text-primary-purple-bright shadow-glow">
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                </div>
                                <div className="flex-1">
                                    <p className="text-sm font-bold text-white text-glow-purple">Speech Recognition</p>
                                    <p className="text-xs text-primary-purple-bright">In Progress</p>
                                </div>
                            </div>

                            {/* Step 3: Queued */}
                            <div className="flex items-center gap-4 opacity-50">
                                <div className="flex size-10 items-center justify-center rounded-full bg-white/5 text-slate-500">
                                    <Clock className="w-5 h-5" />
                                </div>
                                <div className="flex-1">
                                    <p className="text-sm font-medium text-white">Timestamp Sync</p>
                                    <p className="text-xs text-slate-400">Queued</p>
                                </div>
                            </div>

                            {/* Step 4: Queued */}
                            <div className="flex items-center gap-4 opacity-50">
                                <div className="flex size-10 items-center justify-center rounded-full bg-white/5 text-slate-500">
                                    <Clock className="w-5 h-5" />
                                </div>
                                <div className="flex-1">
                                    <p className="text-sm font-medium text-white">Format Export</p>
                                    <p className="text-xs text-slate-400">Queued</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="flex flex-col gap-6">
                    {/* Status Overview */}
                    <div className="glass-panel p-6">
                        <h3 className="text-lg font-bold text-white mb-4">Status Overview</h3>
                        <div className="flex flex-col gap-3">
                            <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/5">
                                <p className="text-sm text-slate-400">Estimated Time</p>
                                <p className="text-sm font-bold text-white">~8 minutes</p>
                            </div>
                            <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/5">
                                <p className="text-sm text-slate-400">Detected Language</p>
                                <p className="text-sm font-bold text-white">English</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

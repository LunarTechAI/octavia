"use client";

import { motion } from "framer-motion";
import { CheckCircle, Loader2, Clock, PlayCircle, ChevronDown, Terminal } from "lucide-react";
import { useState } from "react";

export default function TranslationProgressPage() {
    const [isLogsOpen, setIsLogsOpen] = useState(false);

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 pb-6">
                <div className="flex flex-col gap-1">
                    <h1 className="font-display text-3xl font-black text-white text-glow-purple">Translating: Video_File_Name.mp4</h1>
                    <p className="text-slate-400 text-sm">Real-time AI orchestration in progress. You can monitor the status below.</p>
                </div>
                <div className="flex items-center gap-3">
                    <button className="flex h-10 items-center justify-center rounded-lg bg-white/5 border border-white/10 px-4 text-sm font-bold text-white hover:bg-white/10 transition-colors">
                        Pause
                    </button>
                    <button className="flex h-10 items-center justify-center rounded-lg bg-white/5 border border-white/10 px-4 text-sm font-bold text-white hover:bg-white/10 transition-colors">
                        Resume
                    </button>
                    <button className="flex h-10 items-center justify-center rounded-lg bg-red-500/10 border border-red-500/30 px-4 text-sm font-bold text-red-400 hover:bg-red-500/20 hover:border-red-500/50 transition-colors">
                        Cancel
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div className="lg:col-span-2 flex flex-col gap-8">
                    {/* Overall Progress */}
                    <div className="glass-panel p-6">
                        <div className="flex items-center justify-between gap-6 mb-3">
                            <p className="text-base font-medium text-white">Overall Progress</p>
                            <p className="text-2xl font-bold text-primary-purple-bright text-glow-purple">75%</p>
                        </div>
                        <div className="w-full bg-white/5 rounded-full h-2.5 mb-3 overflow-hidden">
                            <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: "75%" }}
                                transition={{ duration: 1, ease: "easeOut" }}
                                className="bg-primary-purple h-2.5 rounded-full shadow-glow"
                            />
                        </div>
                        <p className="text-sm text-slate-400">Dubbing chunk 56 of 82: Aligning audio with video.</p>
                    </div>

                    {/* Translation Pipeline */}
                    <div>
                        <h2 className="text-xl font-bold text-white mb-4">Translation Pipeline</h2>
                        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
                            {/* Step 1: Completed */}
                            <div className="glass-card flex flex-col items-center gap-3 p-4 text-center border-green-500/30 bg-green-500/5">
                                <div className="flex size-10 items-center justify-center rounded-full bg-green-500/20 text-green-400 shadow-glow">
                                    <CheckCircle className="w-5 h-5" />
                                </div>
                                <div>
                                    <p className="text-sm font-medium text-white">Splitting</p>
                                    <p className="text-xs text-slate-400">Completed</p>
                                </div>
                            </div>

                            {/* Step 2: Completed */}
                            <div className="glass-card flex flex-col items-center gap-3 p-4 text-center border-green-500/30 bg-green-500/5">
                                <div className="flex size-10 items-center justify-center rounded-full bg-green-500/20 text-green-400 shadow-glow">
                                    <CheckCircle className="w-5 h-5" />
                                </div>
                                <div>
                                    <p className="text-sm font-medium text-white">Transcribing</p>
                                    <p className="text-xs text-slate-400">Completed</p>
                                </div>
                            </div>

                            {/* Step 3: Completed */}
                            <div className="glass-card flex flex-col items-center gap-3 p-4 text-center border-green-500/30 bg-green-500/5">
                                <div className="flex size-10 items-center justify-center rounded-full bg-green-500/20 text-green-400 shadow-glow">
                                    <CheckCircle className="w-5 h-5" />
                                </div>
                                <div>
                                    <p className="text-sm font-medium text-white">Translating</p>
                                    <p className="text-xs text-slate-400">Completed</p>
                                </div>
                            </div>

                            {/* Step 4: In Progress */}
                            <div className="glass-panel glass-panel-glow flex flex-col items-center gap-3 p-4 text-center ring-1 ring-primary-purple/50 relative overflow-hidden">
                                <div className="glass-shine" />
                                <div className="relative z-10 flex flex-col items-center gap-3">
                                    <div className="flex size-10 items-center justify-center rounded-full bg-primary-purple/20 text-primary-purple-bright shadow-glow">
                                        <Loader2 className="w-5 h-5 animate-spin" />
                                    </div>
                                    <div>
                                        <p className="text-sm font-bold text-white text-glow-purple">Dubbing</p>
                                        <p className="text-xs text-primary-purple-bright">In Progress</p>
                                    </div>
                                </div>
                            </div>

                            {/* Step 5: Queued */}
                            <div className="glass-card flex flex-col items-center gap-3 p-4 text-center opacity-50">
                                <div className="flex size-10 items-center justify-center rounded-full bg-white/5 text-slate-500">
                                    <Clock className="w-5 h-5" />
                                </div>
                                <div>
                                    <p className="text-sm font-medium text-white">Merging</p>
                                    <p className="text-xs text-slate-400">Queued</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="lg:col-span-1 flex flex-col gap-6">
                    {/* Status Overview */}
                    <div className="glass-panel p-6">
                        <h3 className="text-lg font-bold text-white mb-4">Status Overview</h3>
                        <div className="flex flex-col gap-4">
                            <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/5">
                                <p className="text-sm text-slate-400">Estimated Time</p>
                                <p className="text-sm font-bold text-white">~12 minutes</p>
                            </div>
                            <button className="btn-border-beam w-full group">
                                <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-2.5">
                                    <PlayCircle className="w-5 h-5 group-hover:scale-110 transition-transform" />
                                    <span>Play Sample Chunk</span>
                                </div>
                            </button>
                        </div>
                    </div>

                    {/* Logs */}
                    <div className="glass-panel overflow-hidden">
                        <button
                            onClick={() => setIsLogsOpen(!isLogsOpen)}
                            className="w-full flex items-center justify-between p-4 font-medium text-white bg-white/5 hover:bg-white/10 transition-colors"
                        >
                            <span className="flex items-center gap-2">
                                <Terminal className="w-4 h-4 text-slate-400" />
                                View Technical Logs
                            </span>
                            <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform ${isLogsOpen ? "rotate-180" : ""}`} />
                        </button>

                        {isLogsOpen && (
                            <div className="h-80 overflow-y-auto p-4 font-mono text-xs space-y-1 custom-scrollbar bg-black/20 border-t border-white/5">
                                <p className="text-slate-500">[14:32:01] - <span className="text-slate-300">Starting translation job for Video_File_Name.mp4...</span></p>
                                <p className="text-slate-500">[14:32:02] - <span className="text-slate-300">Video split into 82 chunks.</span></p>
                                <p className="text-slate-500">[14:32:05] - <span className="text-green-400">Transcription successful for chunk 1/82.</span></p>
                                <p className="text-slate-500">[14:32:08] - <span className="text-green-400">Translation successful for chunk 1/82.</span></p>
                                <p className="text-slate-500">[14:32:12] - <span className="text-slate-300">Dubbing chunk 1/82: Aligning audio...</span></p>
                                <p className="text-slate-500">[14:32:15] - <span className="text-green-400">Dubbing successful for chunk 1/82.</span></p>
                                <p className="text-slate-500">[...]</p>
                                <p className="text-slate-500">[14:38:22] - <span className="text-green-400">Transcription successful for chunk 56/82.</span></p>
                                <p className="text-slate-500">[14:38:25] - <span className="text-green-400">Translation successful for chunk 56/82.</span></p>
                                <p className="text-slate-500">[14:38:29] - <span className="text-primary-purple-bright">Dubbing chunk 56/82: API call to TTS initiated.</span></p>
                                <p className="text-slate-500">[14:38:33] - <span className="text-primary-purple-bright">Dubbing chunk 56/82: Received audio stream.</span></p>
                                <p className="text-slate-500">[14:38:35] - <span className="text-primary-purple-bright animate-pulse">Dubbing chunk 56/82: Aligning audio with video.</span></p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

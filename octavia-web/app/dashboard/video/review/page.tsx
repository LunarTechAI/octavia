"use client";

import { motion } from "framer-motion";
import { Play, RefreshCw, Timer, Save, Sparkles, Download } from "lucide-react";

export default function VideoReviewPage() {
    return (
        <div className="space-y-8">
            <div className="flex flex-col lg:flex-row gap-8">
                {/* Left Column: Video Player */}
                <div className="flex-1 flex flex-col gap-4">
                    <div className="glass-panel p-1">
                        <div className="relative flex items-center justify-center bg-black bg-cover bg-center aspect-video rounded-lg overflow-hidden group">
                            {/* Placeholder Background */}
                            <div className="absolute inset-0 bg-gradient-to-br from-purple-900/20 to-black" />

                            <div className="absolute inset-0 bg-black/30 group-hover:bg-black/10 transition-colors" />

                            <button className="relative z-10 flex shrink-0 items-center justify-center rounded-full size-16 bg-white/10 text-white backdrop-blur-md border border-white/20 hover:scale-110 transition-all shadow-glow">
                                <Play className="w-8 h-8 fill-current ml-1" />
                            </button>

                            {/* Controls Overlay */}
                            <div className="absolute inset-x-0 bottom-0 px-4 py-3 bg-gradient-to-t from-black/80 to-transparent">
                                <div className="flex h-1 items-center justify-center gap-0 mb-2 cursor-pointer group/timeline">
                                    <div className="h-1 flex-1 rounded-l-full bg-primary-purple shadow-[0_0_10px_rgba(168,85,247,0.5)]" />
                                    <div className="relative">
                                        <div className="size-3 rounded-full bg-white ring-4 ring-primary-purple/30 scale-0 group-hover/timeline:scale-100 transition-transform" />
                                    </div>
                                    <div className="h-1 flex-[3] rounded-r-full bg-white/20" />
                                </div>
                                <div className="flex items-center justify-between">
                                    <p className="text-white text-xs font-medium tracking-wide">0:37</p>
                                    <p className="text-slate-300 text-xs font-medium tracking-wide">2:23</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Preview Controls */}
                    <div className="flex p-1 bg-white/5 rounded-lg border border-white/10">
                        <label className="flex-1 cursor-pointer">
                            <input type="radio" name="preview-mode" value="full" className="peer sr-only" defaultChecked />
                            <div className="flex items-center justify-center py-2 rounded-md text-sm font-medium text-slate-400 peer-checked:bg-primary-purple/20 peer-checked:text-white peer-checked:shadow-sm transition-all">
                                Full Preview
                            </div>
                        </label>
                        <label className="flex-1 cursor-pointer">
                            <input type="radio" name="preview-mode" value="spot1" className="peer sr-only" />
                            <div className="flex items-center justify-center py-2 rounded-md text-sm font-medium text-slate-400 peer-checked:bg-primary-purple/20 peer-checked:text-white peer-checked:shadow-sm transition-all">
                                Check Spot 1
                            </div>
                        </label>
                        <label className="flex-1 cursor-pointer">
                            <input type="radio" name="preview-mode" value="spot2" className="peer sr-only" />
                            <div className="flex items-center justify-center py-2 rounded-md text-sm font-medium text-slate-400 peer-checked:bg-primary-purple/20 peer-checked:text-white peer-checked:shadow-sm transition-all">
                                Check Spot 2
                            </div>
                        </label>
                    </div>
                </div>

                {/* Right Column: Stats & Actions */}
                <div className="w-full lg:w-96 flex flex-col gap-6">
                    <div className="flex flex-col gap-2">
                        <h1 className="font-display text-3xl font-black text-white text-glow-purple">Your Translation is Ready!</h1>
                        <p className="text-slate-400 text-sm">Review the results and download your video.</p>
                    </div>

                    {/* Stats Grid */}
                    <div className="flex flex-col gap-3">
                        <div className="glass-card p-4 flex items-center gap-4">
                            <div className="flex size-10 items-center justify-center rounded-full bg-green-500/20 text-green-400 shadow-glow">
                                <RefreshCw className="w-5 h-5" />
                            </div>
                            <div>
                                <p className="text-slate-400 text-xs font-medium uppercase tracking-wider">Sync</p>
                                <p className="text-white text-lg font-bold">98% Match</p>
                            </div>
                        </div>
                        <div className="glass-card p-4 flex items-center gap-4">
                            <div className="flex size-10 items-center justify-center rounded-full bg-blue-500/20 text-blue-400 shadow-glow">
                                <Timer className="w-5 h-5" />
                            </div>
                            <div>
                                <p className="text-slate-400 text-xs font-medium uppercase tracking-wider">Duration</p>
                                <p className="text-white text-lg font-bold">Exact Match</p>
                            </div>
                        </div>
                        <div className="glass-card p-4 flex items-center gap-4">
                            <div className="flex size-10 items-center justify-center rounded-full bg-purple-500/20 text-purple-400 shadow-glow">
                                <Save className="w-5 h-5" />
                            </div>
                            <div>
                                <p className="text-slate-400 text-xs font-medium uppercase tracking-wider">Size</p>
                                <p className="text-white text-lg font-bold">Optimized</p>
                            </div>
                        </div>
                    </div>

                    {/* Actions */}
                    <div className="flex flex-col gap-4">
                        <button className="flex items-center justify-center gap-2 w-full py-2.5 rounded-lg border border-primary-purple/30 bg-primary-purple/10 text-primary-purple-bright font-bold hover:bg-primary-purple/20 hover:border-primary-purple/50 transition-all group">
                            <Sparkles className="w-5 h-5 group-hover:rotate-12 transition-transform" />
                            AI Refine (e.g. 'Softer voice?')
                        </button>

                        <div className="pt-4 border-t border-white/10 flex flex-col gap-4">
                            <button className="btn-border-beam w-full group">
                                <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-3">
                                    <Download className="w-5 h-5 group-hover:translate-y-1 transition-transform" />
                                    <span>Download MP4</span>
                                </div>
                            </button>
                            <label className="flex items-center gap-3 cursor-pointer group">
                                <input type="checkbox" className="form-checkbox rounded border-white/20 bg-white/5 text-primary-purple focus:ring-primary-purple focus:ring-offset-0 size-4 group-hover:border-primary-purple/50 transition-colors" />
                                <span className="text-sm text-slate-400 group-hover:text-white transition-colors">Include original files in .zip</span>
                            </label>
                        </div>
                    </div>

                    {/* Feedback */}
                    <div className="pt-4 border-t border-white/10">
                        <p className="text-sm font-medium text-slate-300 mb-3">Help us improve: Pacing Feedback</p>
                        <input type="range" min="1" max="100" defaultValue="50" className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-primary-purple" />
                        <div className="flex justify-between text-xs text-slate-500 mt-2">
                            <span>Too Slow</span>
                            <span>Too Fast</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

"use client";

import { motion } from "framer-motion";
import { Play, Pause, Volume2 } from "lucide-react";
import { useState } from "react";

export function LiveDemo() {
    const [isPlaying, setIsPlaying] = useState(false);
    const [activeTab, setActiveTab] = useState<"original" | "dubbed">("original");

    return (
        <section className="py-24 relative">
            <div className="container mx-auto px-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">

                    {/* Text Content */}
                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                    >
                        <div className="inline-block px-3 py-1 rounded-full bg-accent-pink/10 border border-accent-pink/20 text-accent-pink text-sm font-medium mb-6">
                            Interactive Demo
                        </div>
                        <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">
                            Hear the <span className="text-accent-pink">difference</span>.
                        </h2>
                        <p className="text-gray-400 text-lg mb-8 leading-relaxed">
                            Experience the quality of Octavia's voice cloning and lip-sync technology.
                            Switch between the original English video and the AI-dubbed Spanish version
                            to see the magic in action.
                        </p>

                        <div className="flex flex-col gap-4">
                            <div className="flex items-start gap-4">
                                <div className="w-8 h-8 rounded-full bg-primary-purple/20 flex items-center justify-center mt-1">
                                    <span className="text-primary-purple font-bold">1</span>
                                </div>
                                <div>
                                    <h4 className="text-white font-bold mb-1">Natural Prosody</h4>
                                    <p className="text-sm text-gray-400">The AI understands context and emotion, preserving the original performance.</p>
                                </div>
                            </div>
                            <div className="flex items-start gap-4">
                                <div className="w-8 h-8 rounded-full bg-accent-cyan/20 flex items-center justify-center mt-1">
                                    <span className="text-accent-cyan font-bold">2</span>
                                </div>
                                <div>
                                    <h4 className="text-white font-bold mb-1">Perfect Lip Sync</h4>
                                    <p className="text-sm text-gray-400">Video frames are regenerated to match the new spoken language.</p>
                                </div>
                            </div>
                        </div>
                    </motion.div>

                    {/* Video Player UI */}
                    <motion.div
                        initial={{ opacity: 0, x: 20 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                        className="relative"
                    >
                        {/* Glow Effect */}
                        <div className="absolute -inset-4 bg-gradient-to-r from-primary-purple to-accent-pink opacity-30 blur-3xl rounded-3xl animate-pulse-slow" />

                        <div className="glass-panel-high p-2 relative z-10 border-glow">
                            {/* Player Window */}
                            <div className="relative aspect-video bg-black rounded-lg overflow-hidden group">
                                {/* Placeholder Image/Gradient */}
                                <div className="absolute inset-0 bg-gradient-to-br from-gray-900 to-black flex items-center justify-center">
                                    <div className="text-center">
                                        <div className="w-16 h-16 rounded-full bg-white/10 backdrop-blur-md flex items-center justify-center mx-auto mb-4 cursor-pointer hover:scale-110 transition-transform" onClick={() => setIsPlaying(!isPlaying)}>
                                            {isPlaying ? <Pause className="fill-white text-white" /> : <Play className="fill-white text-white ml-1" />}
                                        </div>
                                        <p className="text-sm text-gray-500 font-mono">DEMO_VIDEO_01.mp4</p>
                                    </div>
                                </div>

                                {/* Controls Overlay */}
                                <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
                                    <div className="flex items-center justify-between text-white">
                                        <div className="flex items-center gap-4">
                                            <button onClick={() => setIsPlaying(!isPlaying)}>
                                                {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                                            </button>
                                            <span className="text-xs font-mono">00:12 / 01:45</span>
                                        </div>
                                        <Volume2 className="w-5 h-5" />
                                    </div>
                                    {/* Progress Bar */}
                                    <div className="mt-3 h-1 bg-white/20 rounded-full overflow-hidden">
                                        <div className="h-full w-1/3 bg-primary-purple rounded-full" />
                                    </div>
                                </div>
                            </div>

                            {/* Language Switcher */}
                            <div className="mt-2 p-2 bg-white/5 rounded-lg flex gap-2">
                                <button
                                    onClick={() => setActiveTab("original")}
                                    className={cn(
                                        "flex-1 py-2 text-sm font-medium rounded-md transition-all",
                                        activeTab === "original"
                                            ? "bg-primary-purple text-white shadow-lg"
                                            : "text-gray-400 hover:text-white hover:bg-white/5"
                                    )}
                                >
                                    Original (English)
                                </button>
                                <button
                                    onClick={() => setActiveTab("dubbed")}
                                    className={cn(
                                        "flex-1 py-2 text-sm font-medium rounded-md transition-all",
                                        activeTab === "dubbed"
                                            ? "bg-accent-pink text-white shadow-lg"
                                            : "text-gray-400 hover:text-white hover:bg-white/5"
                                    )}
                                >
                                    Dubbed (Spanish)
                                </button>
                            </div>
                        </div>
                    </motion.div>

                </div>
            </div>
        </section>
    );
}

function cn(...classes: (string | undefined | null | false)[]) {
    return classes.filter(Boolean).join(" ");
}

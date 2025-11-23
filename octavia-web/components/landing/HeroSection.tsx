"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { ArrowRight, Play } from "lucide-react";

export function HeroSection() {
    return (
        <section className="relative min-h-screen flex items-center justify-center pt-20 overflow-hidden">
            {/* Ambient Background Glows (Matching Dashboard) */}
            <div className="glow-purple-strong"
                style={{ width: "600px", height: "600px", position: "absolute", top: "-200px", right: "-100px", zIndex: 0 }} />
            <div className="glow-purple"
                style={{ width: "400px", height: "400px", position: "absolute", bottom: "-100px", left: "100px", zIndex: 0 }} />

            <div className="container mx-auto px-6 relative z-10 text-center">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                >
                    <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 backdrop-blur-md mb-8 hover:bg-white/10 transition-colors cursor-pointer border-glow">
                        <span className="w-2 h-2 rounded-full bg-accent-cyan animate-pulse" />
                        <span className="text-sm font-medium text-gray-300">
                            Octavia is now live
                        </span>
                        <ArrowRight className="w-4 h-4 text-gray-400" />
                    </div>
                </motion.div>

                <motion.h1
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.1, ease: "easeOut" }}
                    className="text-5xl md:text-7xl font-bold tracking-tight text-white mb-6 text-glow-purple"
                >
                    Universal Translation. <br />
                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-purple-bright to-accent-cyan">
                        Zero Barriers.
                    </span>
                </motion.h1>

                <motion.p
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
                    className="text-xl text-gray-400 max-w-2xl mx-auto mb-10 leading-relaxed"
                >
                    The world's most advanced AI dubbing platform. Translate video, audio,
                    and subtitles into 30+ languages with perfect lip-sync and voice
                    cloning.
                </motion.p>

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.3, ease: "easeOut" }}
                    className="flex flex-col sm:flex-row items-center justify-center gap-4"
                >
                    <Link href="/signup" className="btn-border-beam h-[60px]">
                        <div className="btn-border-beam-inner text-lg px-8 flex items-center justify-center">
                            Start Translating Free
                        </div>
                    </Link>

                    <button className="group h-[60px] flex items-center gap-3 px-8 rounded-full bg-white/5 border border-white/10 hover:bg-white/10 transition-all">
                        <div className="w-9 h-9 rounded-full bg-white/10 flex items-center justify-center group-hover:scale-110 transition-transform">
                            <Play className="w-4 h-4 text-white fill-current" />
                        </div>
                        <span className="font-medium text-white">Watch Demo</span>
                    </button>
                </motion.div>

                {/* Stats / Trust */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 1, delay: 0.6 }}
                    className="mt-20 pt-10 border-t border-white/5 grid grid-cols-2 md:grid-cols-4 gap-8"
                >
                    {[
                        { label: "Languages", value: "30+" },
                        { label: "Accuracy", value: "99%" },
                        { label: "Processing Speed", value: "10x" },
                        { label: "Active Users", value: "10k+" },
                    ].map((stat, i) => (
                        <div key={i}>
                            <div className="text-3xl font-bold text-white mb-1">{stat.value}</div>
                            <div className="text-sm text-gray-500 uppercase tracking-wider">
                                {stat.label}
                            </div>
                        </div>
                    ))}
                </motion.div>
            </div>
        </section>
    );
}

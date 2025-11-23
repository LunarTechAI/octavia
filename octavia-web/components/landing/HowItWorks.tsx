"use client";

import { motion } from "framer-motion";
import { Upload, Cpu, Globe2 } from "lucide-react";

const steps = [
    {
        id: "01",
        title: "Upload Content",
        description: "Drag & drop your video or audio files. We support all major formats up to 5GB.",
        icon: Upload,
    },
    {
        id: "02",
        title: "AI Processing",
        description: "Our engine transcribes, translates, clones voice, and syncs lips in minutes.",
        icon: Cpu,
    },
    {
        id: "03",
        title: "Global Release",
        description: "Download your localized assets or stream them directly via our global CDN.",
        icon: Globe2,
    },
];

export function HowItWorks() {
    return (
        <section id="how-it-works" className="py-24 relative overflow-hidden">
            {/* Background Gradient */}
            <div className="absolute top-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-primary-purple/50 to-transparent" />

            <div className="container mx-auto px-6 relative z-10">
                <div className="text-center mb-20">
                    <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">
                        From upload to <span className="text-accent-cyan">worldwide</span> <br />
                        in three simple steps.
                    </h2>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-12 relative">
                    {/* Connecting Line (Desktop) */}
                    <div className="hidden md:block absolute top-12 left-[16%] right-[16%] h-0.5 bg-white/10 z-0">
                        <div className="absolute inset-0 bg-gradient-to-r from-primary-purple to-accent-cyan opacity-30" />
                    </div>

                    {steps.map((step, index) => (
                        <motion.div
                            key={index}
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: index * 0.2 }}
                            className="relative z-10 text-center group"
                        >
                            <div className="w-24 h-24 mx-auto rounded-2xl glass-panel flex items-center justify-center mb-8 group-hover:border-primary-purple/50 transition-colors relative">
                                <div className="absolute -top-3 -right-3 w-8 h-8 rounded-full bg-bg-surface border border-white/10 flex items-center justify-center text-sm font-bold text-primary-purple">
                                    {step.id}
                                </div>
                                <step.icon className="w-10 h-10 text-white group-hover:scale-110 transition-transform duration-300" />
                            </div>

                            <h3 className="text-xl font-bold text-white mb-3">{step.title}</h3>
                            <p className="text-gray-400 max-w-xs mx-auto">{step.description}</p>
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    );
}

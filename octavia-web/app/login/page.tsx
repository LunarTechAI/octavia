"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowLeft, Mail, Lock } from "lucide-react";

export default function LoginPage() {
    return (
        <div className="min-h-screen w-full bg-bg-dark flex items-center justify-center relative overflow-hidden">
            {/* Ambient Background Glows */}
            <div className="glow-purple-strong"
                style={{ width: "600px", height: "600px", position: "absolute", top: "-200px", right: "-100px", zIndex: 0 }} />
            <div className="glow-purple"
                style={{ width: "400px", height: "400px", position: "absolute", bottom: "-100px", left: "100px", zIndex: 0 }} />

            <div className="relative z-10 w-full max-w-md p-6">
                <Link href="/" className="inline-flex items-center gap-2 text-slate-400 hover:text-white mb-8 transition-colors">
                    <ArrowLeft className="w-4 h-4" />
                    Back to Home
                </Link>

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="glass-panel p-8"
                >
                    <div className="text-center mb-8">
                        <div className="w-12 h-12 mx-auto mb-4 relative flex items-center justify-center">
                            <img
                                src="/lunartech_logo_small.png"
                                alt="LunarTech Logo"
                                className="w-full h-full object-contain"
                            />
                            <div className="absolute inset-0 bg-white/30 blur-xl rounded-full opacity-20" />
                        </div>
                        <h1 className="text-2xl font-bold text-white mb-2">Welcome Back</h1>
                        <p className="text-slate-400 text-sm">Sign in to your Octavia account</p>
                    </div>

                    <form className="space-y-4">
                        <div className="space-y-2">
                            <label className="text-sm font-medium text-slate-300">Email</label>
                            <div className="relative">
                                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                                <input
                                    type="email"
                                    placeholder="name@example.com"
                                    className="glass-input w-full !pl-12"
                                />
                            </div>
                        </div>
                        <div className="space-y-2">
                            <label className="text-sm font-medium text-slate-300">Password</label>
                            <div className="relative">
                                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                                <input
                                    type="password"
                                    placeholder="••••••••"
                                    className="glass-input w-full !pl-12"
                                />
                            </div>
                        </div>

                        <button className="w-full btn-border-beam mt-6">
                            <div className="btn-border-beam-inner justify-center py-3">
                                Sign In
                            </div>
                        </button>
                    </form>

                    <div className="mt-6">
                        <div className="relative">
                            <div className="absolute inset-0 flex items-center">
                                <div className="w-full border-t border-white/10"></div>
                            </div>
                            <div className="relative flex justify-center text-xs uppercase">
                                <span className="bg-[#0D0221] px-2 text-slate-500">Or continue with</span>
                            </div>
                        </div>

                        <div className="mt-6 grid grid-cols-2 gap-3">
                            <button className="flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 transition-colors text-sm text-white">
                                <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
                                    <path d="M17.05 20.28c-.98.95-2.05.8-3.08.35-1.09-.46-2.09-.48-3.24 0-1.44.62-2.2.44-3.06-.35C2.79 15.25 3.51 7.59 9.05 7.31c1.35.07 2.29.74 3.08.74 1.18 0 2.21-.93 3.69-.93.95 0 2.58.5 3.63 1.62-3.28 1.66-2.57 6.62 1.3 8.21-.63 1.72-1.62 3.45-3.7 3.33zM12.03 7.25c-.15-2.23 1.66-4.07 3.74-4.25.29 2.58-2.34 4.5-3.74 4.25z" />
                                </svg>
                                Apple
                            </button>
                            <button className="flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 transition-colors text-sm text-white">
                                <span className="font-bold">G</span>
                                Google
                            </button>
                        </div>
                    </div>

                    <p className="mt-8 text-center text-sm text-slate-400">
                        Don't have an account?{" "}
                        <Link href="/signup" className="text-primary-purple-bright hover:text-white transition-colors font-medium">
                            Sign up
                        </Link>
                    </p>
                </motion.div>
            </div>
        </div>
    );
}

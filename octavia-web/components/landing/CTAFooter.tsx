"use client";

import Link from "next/link";
import { Twitter, Github, Linkedin, Mail } from "lucide-react";

export function CTAFooter() {
    return (
        <footer className="relative pt-24 pb-12 overflow-hidden">
            {/* Background Glow */}
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[400px] bg-primary-purple/10 blur-[120px] rounded-full pointer-events-none" />

            <div className="container mx-auto px-6 relative z-10">
                {/* Main CTA */}
                <div className="text-center mb-20">
                    <h2 className="text-4xl md:text-6xl font-bold text-white mb-8">
                        Ready to break language barriers?
                    </h2>
                    <p className="text-xl text-gray-400 mb-10 max-w-2xl mx-auto">
                        Join thousands of creators and companies using Octavia to reach a global audience.
                    </p>
                    <Link href="/signup" className="btn-border-beam scale-125">
                        <div className="btn-border-beam-inner px-10 py-5 text-xl">
                            Get Started for Free
                        </div>
                    </Link>
                </div>

                {/* Footer Links */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-12 mb-16 border-t border-white/5 pt-16">
                    <div>
                        <div className="flex items-center gap-2 mb-6">
                            <div className="w-6 h-6 rounded bg-gradient-to-br from-primary-purple to-accent-cyan flex items-center justify-center text-white text-xs font-bold">
                                O
                            </div>
                            <span className="text-lg font-bold text-white">Octavia</span>
                        </div>
                        <p className="text-gray-500 text-sm leading-relaxed">
                            The world's most advanced AI dubbing and translation platform.
                        </p>
                    </div>

                    <div>
                        <h4 className="text-white font-bold mb-6">Product</h4>
                        <ul className="space-y-4 text-sm text-gray-400">
                            <li><Link href="#" className="hover:text-primary-purple transition-colors">Features</Link></li>
                            <li><Link href="#" className="hover:text-primary-purple transition-colors">Pricing</Link></li>
                            <li><Link href="#" className="hover:text-primary-purple transition-colors">API</Link></li>
                            <li><Link href="#" className="hover:text-primary-purple transition-colors">Showcase</Link></li>
                        </ul>
                    </div>

                    <div>
                        <h4 className="text-white font-bold mb-6">Company</h4>
                        <ul className="space-y-4 text-sm text-gray-400">
                            <li><Link href="#" className="hover:text-primary-purple transition-colors">About</Link></li>
                            <li><Link href="#" className="hover:text-primary-purple transition-colors">Blog</Link></li>
                            <li><Link href="#" className="hover:text-primary-purple transition-colors">Careers</Link></li>
                            <li><Link href="#" className="hover:text-primary-purple transition-colors">Contact</Link></li>
                        </ul>
                    </div>

                    <div>
                        <h4 className="text-white font-bold mb-6">Connect</h4>
                        <div className="flex gap-4">
                            <Link href="#" className="w-10 h-10 rounded-full bg-white/5 flex items-center justify-center text-gray-400 hover:bg-white/10 hover:text-white transition-all">
                                <Twitter className="w-5 h-5" />
                            </Link>
                            <Link href="#" className="w-10 h-10 rounded-full bg-white/5 flex items-center justify-center text-gray-400 hover:bg-white/10 hover:text-white transition-all">
                                <Github className="w-5 h-5" />
                            </Link>
                            <Link href="#" className="w-10 h-10 rounded-full bg-white/5 flex items-center justify-center text-gray-400 hover:bg-white/10 hover:text-white transition-all">
                                <Linkedin className="w-5 h-5" />
                            </Link>
                        </div>
                    </div>
                </div>

                <div className="text-center text-gray-600 text-sm">
                    © 2025 Octavia AI. All rights reserved.
                </div>
            </div>
        </footer>
    );
}

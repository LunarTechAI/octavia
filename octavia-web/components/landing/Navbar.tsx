"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { Menu, X } from "lucide-react";

export function Navbar() {
    const [isScrolled, setIsScrolled] = useState(false);
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

    useEffect(() => {
        const handleScroll = () => {
            setIsScrolled(window.scrollY > 20);
        };
        window.addEventListener("scroll", handleScroll);
        return () => window.removeEventListener("scroll", handleScroll);
    }, []);

    return (
        <nav
            className={cn(
                "fixed top-0 left-0 right-0 z-50 transition-all duration-300 border-b border-transparent",
                isScrolled
                    ? "bg-[#0D0221]/80 backdrop-blur-xl border-white/10 py-4 shadow-glass"
                    : "bg-transparent py-6"
            )}
        >
            <div className="container mx-auto px-6 flex items-center justify-between">
                {/* Logo */}
                {/* Logo */}
                <Link href="/" className="flex items-center gap-4 group cursor-pointer select-none h-10">
                    <div className="relative w-8 h-8 flex items-center justify-center shrink-0">
                        <motion.div
                            className="w-full h-full relative z-10"
                            whileHover={{ scale: 1.1 }}
                            transition={{ type: "spring", stiffness: 400, damping: 10 }}
                        >
                            <img
                                src="/lunartech_logo_small.png"
                                alt="LunarTech Logo"
                                className="w-full h-full object-contain"
                            />
                        </motion.div>
                        {/* Ambient Pulse Glow */}
                        <motion.div
                            className="absolute inset-0 bg-primary-purple/40 blur-xl rounded-full"
                            animate={{
                                opacity: [0.2, 0.5, 0.2],
                                scale: [0.8, 1.2, 0.8],
                            }}
                            transition={{
                                duration: 3,
                                repeat: Infinity,
                                ease: "easeInOut",
                            }}
                        />
                        {/* Hover Burst Glow */}
                        <div className="absolute inset-0 bg-white/40 blur-xl rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                    </div>
                    <div className="relative flex flex-col justify-center h-full">
                        <span className="font-bold text-white text-lg tracking-[0.25em] leading-none group-hover:text-purple-300 transition-all duration-300 group-hover:-translate-y-2">
                            OCTAVIA
                        </span>
                        <span className="absolute left-0 bottom-1 text-[8px] text-slate-500 tracking-[0.1em] uppercase opacity-0 translate-y-2 group-hover:opacity-100 group-hover:translate-y-0 transition-all duration-300 whitespace-nowrap">
                            Rise Beyond Language
                        </span>
                    </div>
                </Link>

                {/* Desktop Nav */}
                <div className="hidden md:flex items-center gap-8">
                    <NavLink href="#features">Features</NavLink>
                    <NavLink href="#how-it-works">How it Works</NavLink>
                    <NavLink href="#pricing">Pricing</NavLink>
                </div>

                {/* CTA Buttons */}
                <div className="hidden md:flex items-center gap-4">
                    <Link
                        href="/login"
                        className="text-sm font-medium text-gray-300 hover:text-white transition-colors"
                    >
                        Log in
                    </Link>
                    <Link href="/signup" className="btn-border-beam">
                        <div className="btn-border-beam-inner">Get Started</div>
                    </Link>
                </div>

                {/* Mobile Menu Toggle */}
                <button
                    className="md:hidden text-white"
                    onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                >
                    {isMobileMenuOpen ? <X /> : <Menu />}
                </button>
            </div>

            {/* Mobile Menu */}
            {isMobileMenuOpen && (
                <div className="md:hidden absolute top-full left-0 right-0 bg-bg-dark/95 backdrop-blur-xl border-b border-white/10 p-6 flex flex-col gap-4 animate-in slide-in-from-top-5">
                    <MobileNavLink href="#features" onClick={() => setIsMobileMenuOpen(false)}>
                        Features
                    </MobileNavLink>
                    <MobileNavLink href="#how-it-works" onClick={() => setIsMobileMenuOpen(false)}>
                        How it Works
                    </MobileNavLink>
                    <MobileNavLink href="#pricing" onClick={() => setIsMobileMenuOpen(false)}>
                        Pricing
                    </MobileNavLink>
                    <div className="h-px bg-white/10 my-2" />
                    <Link
                        href="/login"
                        className="text-center py-2 text-gray-300 hover:text-white"
                    >
                        Log in
                    </Link>
                    <Link href="/signup" className="btn-border-beam w-full justify-center">
                        <div className="btn-border-beam-inner w-full">Get Started</div>
                    </Link>
                </div>
            )}
        </nav>
    );
}

function NavLink({ href, children }: { href: string; children: React.ReactNode }) {
    return (
        <Link
            href={href}
            className="text-sm font-medium text-gray-400 hover:text-white transition-colors relative group"
        >
            {children}
            <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-primary-purple transition-all group-hover:w-full" />
        </Link>
    );
}

function MobileNavLink({
    href,
    onClick,
    children,
}: {
    href: string;
    onClick: () => void;
    children: React.ReactNode;
}) {
    return (
        <Link
            href={href}
            onClick={onClick}
            className="text-lg font-medium text-gray-300 hover:text-white py-2 block"
        >
            {children}
        </Link>
    );
}

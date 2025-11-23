"use client";

import { motion } from "framer-motion";
import { Check, CreditCard, Zap, Download } from "lucide-react";

const invoices = [
    { id: "INV-001", date: "Oct 23, 2025", amount: "$49.00", status: "Paid" },
    { id: "INV-002", date: "Sep 23, 2025", amount: "$49.00", status: "Paid" },
    { id: "INV-003", date: "Aug 23, 2025", amount: "$49.00", status: "Paid" },
];

export default function BillingPage() {
    return (
        <div className="space-y-8">
            {/* Header */}
            <div>
                <h1 className="font-display text-3xl font-black text-white mb-2 text-glow-purple">Plans & Billing</h1>
                <p className="text-slate-400 text-sm">Manage your subscription and payment methods</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Current Plan */}
                <div className="lg:col-span-2 space-y-6">
                    <div className="glass-panel p-8 relative overflow-hidden">
                        <div className="glass-shine" />
                        <div className="relative z-10 flex flex-col md:flex-row justify-between gap-6">
                            <div>
                                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary-purple/20 border border-primary-purple/30 text-primary-purple-bright text-xs font-bold uppercase tracking-wider mb-4">
                                    Current Plan
                                </div>
                                <h2 className="text-3xl font-bold text-white mb-2">Pro Plan</h2>
                                <p className="text-slate-400 max-w-md">
                                    You are on the Pro plan. Your next billing date is <span className="text-white font-medium">November 23, 2025</span>.
                                </p>
                            </div>
                            <div className="flex flex-col items-start md:items-end gap-3">
                                <span className="text-4xl font-bold text-white">$49<span className="text-lg text-slate-500 font-normal">/mo</span></span>
                                <button className="btn-border-beam">
                                    <div className="btn-border-beam-inner px-6 py-2">
                                        Manage Subscription
                                    </div>
                                </button>
                            </div>
                        </div>

                        {/* Usage Stats */}
                        <div className="mt-8 grid grid-cols-1 sm:grid-cols-3 gap-4 pt-8 border-t border-white/10">
                            <div>
                                <div className="text-slate-500 text-xs uppercase tracking-wider mb-1">Translation Minutes</div>
                                <div className="text-2xl font-bold text-white">450 <span className="text-sm text-slate-500 font-normal">/ 1000</span></div>
                                <div className="w-full h-1 bg-white/10 rounded-full mt-2 overflow-hidden">
                                    <div className="h-full bg-primary-purple w-[45%]" />
                                </div>
                            </div>
                            <div>
                                <div className="text-slate-500 text-xs uppercase tracking-wider mb-1">Storage Used</div>
                                <div className="text-2xl font-bold text-white">12.5 <span className="text-sm text-slate-500 font-normal">/ 50 GB</span></div>
                                <div className="w-full h-1 bg-white/10 rounded-full mt-2 overflow-hidden">
                                    <div className="h-full bg-accent-cyan w-[25%]" />
                                </div>
                            </div>
                            <div>
                                <div className="text-slate-500 text-xs uppercase tracking-wider mb-1">Active Projects</div>
                                <div className="text-2xl font-bold text-white">8 <span className="text-sm text-slate-500 font-normal">/ Unlimited</span></div>
                            </div>
                        </div>
                    </div>

                    {/* Payment Method */}
                    <div className="glass-panel p-6">
                        <h3 className="text-white font-bold text-lg mb-4">Payment Method</h3>
                        <div className="flex items-center justify-between p-4 rounded-xl bg-white/5 border border-white/10">
                            <div className="flex items-center gap-4">
                                <div className="w-12 h-8 rounded bg-white/10 flex items-center justify-center border border-white/10">
                                    <div className="w-6 h-4 bg-white/80 rounded-sm" />
                                </div>
                                <div>
                                    <div className="text-white font-medium">Visa ending in 4242</div>
                                    <div className="text-slate-500 text-xs">Expires 12/28</div>
                                </div>
                            </div>
                            <button className="text-sm text-primary-purple-bright hover:text-white transition-colors">
                                Edit
                            </button>
                        </div>
                    </div>
                </div>

                {/* Invoice History */}
                <div className="glass-panel p-6 h-fit">
                    <h3 className="text-white font-bold text-lg mb-4">Invoice History</h3>
                    <div className="space-y-1">
                        {invoices.map((invoice) => (
                            <div key={invoice.id} className="flex items-center justify-between p-3 rounded-lg hover:bg-white/5 transition-colors group cursor-pointer">
                                <div>
                                    <div className="text-white font-medium text-sm">{invoice.date}</div>
                                    <div className="text-slate-500 text-xs">{invoice.id} • {invoice.amount}</div>
                                </div>
                                <div className="flex items-center gap-3">
                                    <span className="px-2 py-0.5 rounded-full bg-green-500/10 text-green-400 text-[10px] font-bold uppercase border border-green-500/20">
                                        {invoice.status}
                                    </span>
                                    <Download className="w-4 h-4 text-slate-500 group-hover:text-white transition-colors" />
                                </div>
                            </div>
                        ))}
                    </div>
                    <button className="w-full mt-4 py-2 text-sm text-slate-400 hover:text-white transition-colors border-t border-white/5">
                        View All Invoices
                    </button>
                </div>
            </div>
        </div>
    );
}

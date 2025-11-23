"use client";

import { motion } from "framer-motion";
import { Check } from "lucide-react";
import Link from "next/link";

const plans = [
    {
        name: "Starter",
        price: "Free",
        description: "Perfect for trying out Octavia.",
        features: ["10 minutes / month", "720p export", "Standard voices", "Watermarked"],
        cta: "Start Free",
        highlight: false,
    },
    {
        name: "Pro",
        price: "$29",
        period: "/mo",
        description: "For creators and small teams.",
        features: ["120 minutes / month", "4K export", "Ultra-realistic voices", "No watermark", "Priority support"],
        cta: "Get Pro",
        highlight: true,
    },
    {
        name: "Enterprise",
        price: "Custom",
        description: "For large scale media companies.",
        features: ["Unlimited minutes", "API Access", "Custom voice clones", "SLA & Dedicated Manager", "SSO"],
        cta: "Contact Sales",
        highlight: false,
    },
];

export function PricingPreview() {
    return (
        <section id="pricing" className="py-24 relative">
            <div className="container mx-auto px-6">
                <div className="text-center mb-16">
                    <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">
                        Simple, transparent <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-primary-purple">pricing</span>.
                    </h2>
                    <p className="text-gray-400">
                        Start for free, upgrade as you grow. No hidden fees.
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
                    {plans.map((plan, index) => (
                        <motion.div
                            key={index}
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: index * 0.1 }}
                            className={`relative glass-card flex flex-col !overflow-visible ${plan.highlight ? "border-primary-purple shadow-glow" : ""
                                }`}
                        >
                            {plan.highlight && (
                                <div className="absolute -top-4 left-1/2 -translate-x-1/2 px-4 py-1 rounded-full bg-gradient-to-r from-blue-600 to-primary-purple text-white text-sm font-bold shadow-lg whitespace-nowrap">
                                    Most Popular
                                </div>
                            )}

                            <div className="mb-8">
                                <h3 className="text-xl font-bold text-white mb-2">{plan.name}</h3>
                                <div className="flex items-baseline gap-1">
                                    <span className="text-4xl font-bold text-white">{plan.price}</span>
                                    {plan.period && <span className="text-gray-400">{plan.period}</span>}
                                </div>
                                <p className="text-gray-400 mt-4 text-sm">{plan.description}</p>
                            </div>

                            <div className="flex-1 mb-8">
                                <ul className="space-y-4">
                                    {plan.features.map((feature, i) => (
                                        <li key={i} className="flex items-center gap-3 text-gray-300 text-sm">
                                            <div className={`w-5 h-5 rounded-full flex items-center justify-center ${plan.highlight ? "bg-blue-500/20 text-blue-400" : "bg-white/10 text-white"}`}>
                                                <Check className="w-3 h-3" />
                                            </div>
                                            {feature}
                                        </li>
                                    ))}
                                </ul>
                            </div>

                            <Link
                                href="/signup"
                                className={`w-full py-3 rounded-lg font-bold text-center transition-all ${plan.highlight
                                    ? "bg-gradient-to-r from-blue-600 to-primary-purple hover:from-blue-500 hover:to-primary-purple-bright text-white shadow-lg shadow-primary-purple/25"
                                    : "bg-white/5 hover:bg-white/10 text-white border border-white/10"
                                    }`}
                            >
                                {plan.cta}
                            </Link>
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    );
}

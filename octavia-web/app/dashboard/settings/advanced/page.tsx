"use client";

import { useState } from "react";
import { Sparkles, Settings as SettingsIcon, Zap, Database, Save } from "lucide-react";

export default function AdvancedSettingsPage() {
    const [activeTab, setActiveTab] = useState("magic");

    return (
        <div className="space-y-8">
            {/* Header */}
            <div>
                <h1 className="font-display text-3xl font-black text-white text-glow-purple mb-2">Advanced Settings</h1>
                <p className="text-slate-400 text-sm">Configure advanced features and AI behavior</p>
            </div>

            {/* Tabs */}
            <div className="flex gap-2 p-1 bg-white/5 rounded-lg border border-white/10">
                <button
                    onClick={() => setActiveTab("magic")}
                    className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-md text-sm font-medium transition-all ${activeTab === "magic"
                            ? "bg-primary-purple/20 text-white shadow-sm"
                            : "text-slate-400 hover:text-white"
                        }`}
                >
                    <Sparkles className="w-4 h-4" />
                    Magic Mode
                </button>
                <button
                    onClick={() => setActiveTab("performance")}
                    className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-md text-sm font-medium transition-all ${activeTab === "performance"
                            ? "bg-primary-purple/20 text-white shadow-sm"
                            : "text-slate-400 hover:text-white"
                        }`}
                >
                    <Zap className="w-4 h-4" />
                    Performance
                </button>
                <button
                    onClick={() => setActiveTab("data")}
                    className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-md text-sm font-medium transition-all ${activeTab === "data"
                            ? "bg-primary-purple/20 text-white shadow-sm"
                            : "text-slate-400 hover:text-white"
                        }`}
                >
                    <Database className="w-4 h-4" />
                    Data & Storage
                </button>
            </div>

            {/* Tab Content */}
            <div className="space-y-6">
                {activeTab === "magic" && (
                    <>
                        <div className="glass-panel p-6">
                            <h2 className="text-xl font-bold text-white mb-4">Magic Mode Features</h2>
                            <div className="space-y-4">
                                <label className="flex items-center justify-between cursor-pointer">
                                    <div>
                                        <p className="text-white font-medium">Smart Voice Cloning</p>
                                        <p className="text-slate-400 text-xs">Automatically clone voices from source media</p>
                                    </div>
                                    <input type="checkbox" className="form-checkbox rounded border-white/20 bg-white/5 text-primary-purple focus:ring-primary-purple size-5" defaultChecked />
                                </label>

                                <label className="flex items-center justify-between cursor-pointer">
                                    <div>
                                        <p className="text-white font-medium">Auto Lip Sync</p>
                                        <p className="text-slate-400 text-xs">Synchronize lip movements with translated audio</p>
                                    </div>
                                    <input type="checkbox" className="form-checkbox rounded border-white/20 bg-white/5 text-primary-purple focus:ring-primary-purple size-5" defaultChecked />
                                </label>

                                <label className="flex items-center justify-between cursor-pointer">
                                    <div>
                                        <p className="text-white font-medium">Context-Aware Translation</p>
                                        <p className="text-slate-400 text-xs">Use AI to understand context for better translations</p>
                                    </div>
                                    <input type="checkbox" className="form-checkbox rounded border-white/20 bg-white/5 text-primary-purple focus:ring-primary-purple size-5" defaultChecked />
                                </label>
                            </div>
                        </div>

                        <div className="glass-panel p-6">
                            <h2 className="text-xl font-bold text-white mb-4">AI Model Selection</h2>
                            <div>
                                <label className="text-white text-sm font-semibold mb-2 block">Translation Model</label>
                                <select className="glass-select w-full">
                                    <option>GPT-4 Turbo (Recommended)</option>
                                    <option>GPT-3.5 (Faster)</option>
                                    <option>Claude Sonnet</option>
                                </select>
                            </div>
                        </div>
                    </>
                )}

                {activeTab === "performance" && (
                    <div className="glass-panel p-6">
                        <h2 className="text-xl font-bold text-white mb-4">Performance Settings</h2>
                        <div className="space-y-4">
                            <div>
                                <label className="text-white text-sm font-semibold mb-2 block">Processing Priority</label>
                                <select className="glass-select w-full">
                                    <option>Balanced</option>
                                    <option>Speed</option>
                                    <option>Quality</option>
                                </select>
                            </div>

                            <div>
                                <label className="text-white text-sm font-semibold mb-2 block">Concurrent Jobs</label>
                                <select className="glass-select w-full">
                                    <option>1 (Sequential)</option>
                                    <option>2 (Recommended)</option>
                                    <option>4 (Maximum)</option>
                                </select>
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === "data" && (
                    <div className="glass-panel p-6">
                        <h2 className="text-xl font-bold text-white mb-4">Data & Storage</h2>
                        <div className="space-y-4">
                            <div className="flex items-center justify-between p-4 bg-white/5 rounded-lg">
                                <div>
                                    <p className="text-white font-medium">Total Storage Used</p>
                                    <p className="text-slate-400 text-xs">12.5 GB of 50 GB</p>
                                </div>
                                <div className="text-white text-2xl font-bold">25%</div>
                            </div>

                            <label className="flex items-center justify-between cursor-pointer">
                                <div>
                                    <p className="text-white font-medium">Auto-delete completed jobs</p>
                                    <p className="text-slate-400 text-xs">Remove files after 30 days</p>
                                </div>
                                <input type="checkbox" className="form-checkbox rounded border-white/20 bg-white/5 text-primary-purple focus:ring-primary-purple size-5" />
                            </label>

                            <button className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg border border-red-500/30 bg-red-500/10 text-red-400 font-bold hover:bg-red-500/20 hover:border-red-500/50 transition-all">
                                Clear All Cache
                            </button>
                        </div>
                    </div>
                )}

                {/* Save Button */}
                <button className="btn-border-beam w-full group">
                    <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-3">
                        <Save className="w-5 h-5" />
                        <span>Save Changes</span>
                    </div>
                </button>
            </div>
        </div>
    );
}

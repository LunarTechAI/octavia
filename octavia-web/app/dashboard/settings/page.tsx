"use client";

import { Bell, Globe, Shield, Palette, Save } from "lucide-react";

export default function SettingsPage() {
    return (
        <div className="space-y-8">
            {/* Header */}
            <div>
                <h1 className="font-display text-3xl font-black text-white text-glow-purple mb-2">Settings</h1>
                <p className="text-slate-400 text-sm">Manage your account preferences and application settings</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Left: Settings Navigation */}
                <div className="space-y-3">
                    <button className="w-full flex items-center gap-3 px-4 py-3 rounded-lg bg-primary-purple/10 border border-primary-purple/30 text-primary-purple-bright transition-all">
                        <Bell className="w-5 h-5" />
                        <span className="text-sm font-medium">Notifications</span>
                    </button>
                    <button className="w-full flex items-center gap-3 px-4 py-3 rounded-lg bg-white/5 border border-white/10 text-slate-400 hover:text-white hover:bg-white/10 transition-all">
                        <Globe className="w-5 h-5" />
                        <span className="text-sm font-medium">Language & Region</span>
                    </button>
                    <button className="w-full flex items-center gap-3 px-4 py-3 rounded-lg bg-white/5 border border-white/10 text-slate-400 hover:text-white hover:bg-white/10 transition-all">
                        <Palette className="w-5 h-5" />
                        <span className="text-sm font-medium">Appearance</span>
                    </button>
                    <button className="w-full flex items-center gap-3 px-4 py-3 rounded-lg bg-white/5 border border-white/10 text-slate-400 hover:text-white hover:bg-white/10 transition-all">
                        <Shield className="w-5 h-5" />
                        <span className="text-sm font-medium">Privacy</span>
                    </button>
                </div>

                {/* Right: Settings Content */}
                <div className="lg:col-span-2 space-y-6">
                    {/* Notifications Section */}
                    <div className="glass-panel p-6">
                        <h2 className="text-xl font-bold text-white mb-4">Notification Preferences</h2>
                        <div className="space-y-4">
                            <label className="flex items-center justify-between cursor-pointer group">
                                <div>
                                    <p className="text-white font-medium">Translation Complete</p>
                                    <p className="text-slate-400 text-xs">Receive notifications when translations finish</p>
                                </div>
                                <input type="checkbox" className="form-checkbox rounded border-white/20 bg-white/5 text-primary-purple focus:ring-primary-purple focus:ring-offset-0 size-5" defaultChecked />
                            </label>

                            <label className="flex items-center justify-between cursor-pointer group">
                                <div>
                                    <p className="text-white font-medium">Email Notifications</p>
                                    <p className="text-slate-400 text-xs">Send email updates for completed jobs</p>
                                </div>
                                <input type="checkbox" className="form-checkbox rounded border-white/20 bg-white/5 text-primary-purple focus:ring-primary-purple focus:ring-offset-0 size-5" />
                            </label>

                            <label className="flex items-center justify-between cursor-pointer group">
                                <div>
                                    <p className="text-white font-medium">Weekly Summary</p>
                                    <p className="text-slate-400 text-xs">Receive a weekly report of your activity</p>
                                </div>
                                <input type="checkbox" className="form-checkbox rounded border-white/20 bg-white/5 text-primary-purple focus:ring-primary-purple focus:ring-offset-0 size-5" defaultChecked />
                            </label>
                        </div>
                    </div>

                    {/* Language & Region */}
                    <div className="glass-panel p-6">
                        <h2 className="text-xl font-bold text-white mb-4">Language & Region</h2>
                        <div className="space-y-4">
                            <div>
                                <label className="text-white text-sm font-semibold mb-2 block">Interface Language</label>
                                <select className="glass-select w-full">
                                    <option>English</option>
                                    <option>Spanish</option>
                                    <option>French</option>
                                    <option>German</option>
                                </select>
                            </div>

                            <div>
                                <label className="text-white text-sm font-semibold mb-2 block">Time Zone</label>
                                <select className="glass-select w-full">
                                    <option>UTC (GMT+0)</option>
                                    <option>EST (GMT-5)</option>
                                    <option>PST (GMT-8)</option>
                                    <option>CET (GMT+1)</option>
                                </select>
                            </div>
                        </div>
                    </div>

                    {/* Save Button */}
                    <button className="btn-border-beam w-full group">
                        <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-3">
                            <Save className="w-5 h-5" />
                            <span>Save Changes</span>
                        </div>
                    </button>
                </div>
            </div>
        </div>
    );
}

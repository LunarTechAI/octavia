"use client";

import { User, Lock, Shield, Mail, Camera } from "lucide-react";

export default function ProfilePage() {
    return (
        <div className="space-y-8">
            {/* Header */}
            <div>
                <h1 className="font-display text-3xl font-black text-white mb-2 text-glow-purple">Profile & Security</h1>
                <p className="text-slate-400 text-sm">Manage your personal information and account security</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Left Column: Personal Info */}
                <div className="lg:col-span-2 space-y-6">
                    {/* Profile Card */}
                    <div className="glass-panel p-8">
                        <div className="flex items-center gap-6 mb-8">
                            <div className="relative group cursor-pointer">
                                <div className="w-24 h-24 rounded-full bg-white/10 border-2 border-white/20 flex items-center justify-center overflow-hidden">
                                    <User className="w-10 h-10 text-slate-400" />
                                </div>
                                <div className="absolute inset-0 bg-black/50 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                                    <Camera className="w-6 h-6 text-white" />
                                </div>
                            </div>
                            <div>
                                <h2 className="text-2xl font-bold text-white">Alex Morgan</h2>
                                <p className="text-slate-400">alex.morgan@example.com</p>
                                <div className="mt-2 inline-flex items-center gap-2 px-2 py-1 rounded-md bg-primary-purple/10 border border-primary-purple/20 text-primary-purple-bright text-xs font-medium">
                                    <Shield className="w-3 h-3" />
                                    Admin
                                </div>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-slate-300">Full Name</label>
                                <input
                                    type="text"
                                    defaultValue="Alex Morgan"
                                    className="glass-input w-full"
                                />
                            </div>
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-slate-300">Email Address</label>
                                <div className="relative">
                                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                                    <input
                                        type="email"
                                        defaultValue="alex.morgan@example.com"
                                        className="glass-input w-full pl-10"
                                    />
                                </div>
                            </div>
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-slate-300">Company</label>
                                <input
                                    type="text"
                                    defaultValue="LunarTech"
                                    className="glass-input w-full"
                                />
                            </div>
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-slate-300">Role</label>
                                <input
                                    type="text"
                                    defaultValue="Product Manager"
                                    className="glass-input w-full"
                                />
                            </div>
                        </div>

                        <div className="mt-8 flex justify-end">
                            <button className="btn-border-beam">
                                <div className="btn-border-beam-inner px-6 py-2">
                                    Save Changes
                                </div>
                            </button>
                        </div>
                    </div>
                </div>

                {/* Right Column: Security */}
                <div className="space-y-6">
                    <div className="glass-panel p-6">
                        <div className="flex items-center gap-3 mb-6">
                            <div className="p-2 rounded-lg bg-primary-purple/10 text-primary-purple-bright">
                                <Lock className="w-5 h-5" />
                            </div>
                            <h3 className="text-white font-bold text-lg">Security</h3>
                        </div>

                        <div className="space-y-6">
                            <div>
                                <h4 className="text-sm font-medium text-white mb-2">Password</h4>
                                <p className="text-xs text-slate-400 mb-3">Last changed 3 months ago</p>
                                <button className="w-full py-2 rounded-lg border border-white/10 hover:bg-white/5 text-sm text-slate-300 hover:text-white transition-colors">
                                    Change Password
                                </button>
                            </div>

                            <div className="pt-6 border-t border-white/5">
                                <div className="flex items-center justify-between mb-2">
                                    <h4 className="text-sm font-medium text-white">Two-Factor Auth</h4>
                                    <span className="text-xs text-green-400 font-medium">Enabled</span>
                                </div>
                                <p className="text-xs text-slate-400 mb-3">Secure your account with 2FA.</p>
                                <button className="w-full py-2 rounded-lg border border-white/10 hover:bg-white/5 text-sm text-slate-300 hover:text-white transition-colors">
                                    Configure
                                </button>
                            </div>
                        </div>
                    </div>

                    <div className="glass-panel p-6 border-red-500/20">
                        <h3 className="text-red-400 font-bold text-lg mb-2">Danger Zone</h3>
                        <p className="text-slate-400 text-xs mb-4">
                            Permanently delete your account and all of your content.
                        </p>
                        <button className="w-full py-2 rounded-lg bg-red-500/10 hover:bg-red-500/20 text-red-400 text-sm font-medium transition-colors border border-red-500/20">
                            Delete Account
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

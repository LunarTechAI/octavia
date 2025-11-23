"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
    LayoutGrid,
    Mic,
    CreditCard,
    History,
    User,
    Users,
    LogOut,
    HelpCircle,
    LifeBuoy,
    Settings,
    Folder
} from "lucide-react";

const navItems = [
    { name: "New Translation", href: "/dashboard", icon: LayoutGrid, primary: true },
    { name: "My Voices", href: "/dashboard/voices", icon: Mic },
    { name: "Projects", href: "/dashboard/projects", icon: Folder },
    { name: "Job History", href: "/dashboard/history", icon: History },
    { name: "Billing", href: "/dashboard/billing", icon: CreditCard },
    { name: "Team", href: "/dashboard/team", icon: Users },
    { name: "Settings", href: "/dashboard/settings", icon: Settings },
];

const bottomItems = [
    { name: "Profile", href: "/dashboard/profile", icon: User },
    { name: "Help", href: "/dashboard/help", icon: HelpCircle },
    { name: "Support", href: "/dashboard/support", icon: LifeBuoy },
];

export function Sidebar() {
    const pathname = usePathname();

    return (
        <div className="flex h-full w-64 flex-col justify-between border-r border-white/10 bg-[#0D0221]/50 backdrop-blur-xl p-4">
            <div className="flex flex-col gap-6">
                {/* Logo Area */}
                <div className="flex items-center gap-3 px-3">
                    <div className="relative w-10 h-10 flex items-center justify-center shrink-0">
                        <img
                            src="/lunartech_logo_small.png"
                            alt="LunarTech Logo"
                            className="w-full h-full object-contain"
                        />
                        <div className="absolute inset-0 bg-white/30 blur-xl rounded-full opacity-20" />
                    </div>
                    <div className="flex flex-col">
                        <h1 className="text-white text-base font-bold leading-normal">Octavia</h1>
                        <p className="text-slate-500 text-xs font-normal leading-normal">AI Translation Suite</p>
                    </div>
                </div>

                {/* Main Nav */}
                <div className="flex flex-col gap-2">
                    {navItems.map((item) => {
                        const isActive = pathname === item.href;
                        const Icon = item.icon;

                        return (
                            <Link
                                key={item.href}
                                href={item.href}
                                className={cn(
                                    "flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200 group",
                                    item.primary
                                        ? "border border-primary-purple/30 bg-primary-purple/10 hover:bg-primary-purple/20"
                                        : "hover:bg-white/5",
                                    isActive ? "bg-white/10 text-white" : "text-slate-400 hover:text-white"
                                )}
                            >
                                <Icon
                                    className={cn(
                                        "w-5 h-5 transition-colors",
                                        item.primary ? "text-primary-purple-bright" : "group-hover:text-white",
                                        isActive ? "text-primary-purple-bright" : "text-slate-500"
                                    )}
                                />
                                <span className={cn(
                                    "text-sm font-medium leading-normal",
                                    item.primary && "text-primary-purple-bright"
                                )}>
                                    {item.name}
                                </span>
                            </Link>
                        );
                    })}
                </div>
            </div>

            {/* Bottom Nav */}
            <div className="flex flex-col gap-1.5 pt-4 border-t border-white/5">
                {bottomItems.map((item) => {
                    const isActive = pathname === item.href;
                    const Icon = item.icon;

                    return (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={cn(
                                "flex items-center gap-3 px-3 py-2 rounded-lg transition-all duration-200 hover:bg-white/5 group",
                                isActive ? "bg-white/10 text-white" : "text-slate-500 hover:text-slate-300"
                            )}
                        >
                            <Icon className={cn(
                                "w-4 h-4 transition-colors",
                                isActive ? "text-white" : "group-hover:text-slate-300"
                            )} />
                            <span className="text-xs font-medium leading-normal">
                                {item.name}
                            </span>
                        </Link>
                    );
                })}

                <button className="flex w-full items-center gap-3 px-3 py-2 text-slate-500 hover:bg-white/5 hover:text-slate-300 rounded-lg transition-all group mt-1">
                    <LogOut className="w-4 h-4 group-hover:text-slate-300" />
                    <span className="text-xs font-medium leading-normal">Logout</span>
                </button>
            </div>
        </div>
    );
}

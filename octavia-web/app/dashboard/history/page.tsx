"use client";

import { motion } from "framer-motion";
import { Search, Filter, Download, ExternalLink, Clock, CheckCircle2, AlertCircle } from "lucide-react";

const jobs = [
    { id: "JOB-1024", name: "Product Demo Video", type: "Video Translation", lang: "EN → ES", status: "Completed", date: "2 hours ago", duration: "4:30" },
    { id: "JOB-1023", name: "Marketing Podcast Ep. 4", type: "Audio Translation", lang: "EN → FR", status: "Processing", date: "5 hours ago", duration: "24:15" },
    { id: "JOB-1022", name: "Tutorial Series - Part 1", type: "Subtitle Gen", lang: "EN", status: "Completed", date: "Yesterday", duration: "12:00" },
    { id: "JOB-1021", name: "Keynote Speech", type: "Video Translation", lang: "EN → DE", status: "Failed", date: "2 days ago", duration: "45:00" },
    { id: "JOB-1020", name: "User Interview", type: "Audio Translation", lang: "EN → IT", status: "Completed", date: "3 days ago", duration: "15:45" },
];

export default function JobHistoryPage() {
    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h1 className="font-display text-3xl font-black text-white mb-2 text-glow-purple">Job History</h1>
                    <p className="text-slate-400 text-sm">View and manage your past translation tasks</p>
                </div>

                <div className="flex items-center gap-3">
                    <div className="relative">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                        <input
                            type="text"
                            placeholder="Search jobs..."
                            className="pl-10 pr-4 py-2 rounded-lg bg-white/5 border border-white/10 text-sm text-white placeholder:text-slate-500 focus:outline-none focus:border-primary-purple/50 w-64 transition-all"
                        />
                    </div>
                    <button className="p-2 rounded-lg bg-white/5 border border-white/10 text-slate-400 hover:text-white hover:bg-white/10 transition-colors">
                        <Filter className="w-4 h-4" />
                    </button>
                </div>
            </div>

            {/* Jobs List */}
            <div className="glass-panel overflow-hidden">
                <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                        <thead>
                            <tr className="border-b border-white/5 bg-white/5">
                                <th className="p-4 text-xs font-semibold text-slate-400 uppercase tracking-wider">Job Details</th>
                                <th className="p-4 text-xs font-semibold text-slate-400 uppercase tracking-wider">Type</th>
                                <th className="p-4 text-xs font-semibold text-slate-400 uppercase tracking-wider">Status</th>
                                <th className="p-4 text-xs font-semibold text-slate-400 uppercase tracking-wider">Date</th>
                                <th className="p-4 text-xs font-semibold text-slate-400 uppercase tracking-wider text-right">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-white/5">
                            {jobs.map((job) => (
                                <tr key={job.id} className="group hover:bg-white/5 transition-colors">
                                    <td className="p-4">
                                        <div className="flex items-center gap-3">
                                            <div className="w-10 h-10 rounded bg-white/5 flex items-center justify-center border border-white/10">
                                                {job.type.includes("Video") ? (
                                                    <span className="text-lg">🎬</span>
                                                ) : job.type.includes("Audio") ? (
                                                    <span className="text-lg">🎙️</span>
                                                ) : (
                                                    <span className="text-lg">📝</span>
                                                )}
                                            </div>
                                            <div>
                                                <div className="font-medium text-white group-hover:text-primary-purple-bright transition-colors">{job.name}</div>
                                                <div className="text-xs text-slate-500">{job.id} • {job.duration}</div>
                                            </div>
                                        </div>
                                    </td>
                                    <td className="p-4">
                                        <div className="text-sm text-slate-300">{job.type}</div>
                                        <div className="text-xs text-slate-500">{job.lang}</div>
                                    </td>
                                    <td className="p-4">
                                        <div className="flex items-center gap-2">
                                            {job.status === "Completed" && <CheckCircle2 className="w-4 h-4 text-green-400" />}
                                            {job.status === "Processing" && <Clock className="w-4 h-4 text-blue-400 animate-pulse" />}
                                            {job.status === "Failed" && <AlertCircle className="w-4 h-4 text-red-400" />}
                                            <span className={`text-sm font-medium ${job.status === "Completed" ? "text-green-400" :
                                                    job.status === "Processing" ? "text-blue-400" : "text-red-400"
                                                }`}>
                                                {job.status}
                                            </span>
                                        </div>
                                    </td>
                                    <td className="p-4 text-sm text-slate-400">{job.date}</td>
                                    <td className="p-4 text-right">
                                        <div className="flex items-center justify-end gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                            <button className="p-2 rounded hover:bg-white/10 text-slate-400 hover:text-white transition-colors" title="Download">
                                                <Download className="w-4 h-4" />
                                            </button>
                                            <button className="p-2 rounded hover:bg-white/10 text-slate-400 hover:text-white transition-colors" title="View Details">
                                                <ExternalLink className="w-4 h-4" />
                                            </button>
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>

                {/* Pagination */}
                <div className="p-4 border-t border-white/5 flex items-center justify-between text-sm text-slate-500">
                    <div>Showing 1-5 of 24 jobs</div>
                    <div className="flex gap-2">
                        <button className="px-3 py-1 rounded hover:bg-white/5 hover:text-white disabled:opacity-50" disabled>Previous</button>
                        <button className="px-3 py-1 rounded hover:bg-white/5 hover:text-white">Next</button>
                    </div>
                </div>
            </div>
        </div>
    );
}

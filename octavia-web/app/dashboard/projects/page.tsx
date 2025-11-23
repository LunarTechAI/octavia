"use client";

import { motion } from "framer-motion";
import { Folder, Plus, Clock, CheckCircle, AlertCircle } from "lucide-react";

const projects = [
    { id: 1, name: "Marketing Video ES-FR", type: "Video Translation", status: "completed", date: "2024-11-20", files: 1 },
    { id: 2, name: "Podcast Series", type: "Audio Translation", status: "in-progress", date: "2024-11-22", files: 8 },
    { id: 3, name: "Tutorial Subtitles", type: "Subtitle Generation", status: "completed", date: "2024-11-18", files: 3 },
    { id: 4, name: "Webinar Recording", type: "Video Translation", status: "pending", date: "2024-11-23", files: 1 },
];

const getStatusBadge = (status: string) => {
    const statusMap: Record<string, { bg: string, text: string, icon: any }> = {
        "completed": { bg: "bg-green-500/10", text: "text-green-400", icon: CheckCircle },
        "in-progress": { bg: "bg-primary-purple/10", text: "text-primary-purple-bright", icon: Clock },
        "pending": { bg: "bg-orange-500/10", text: "text-orange-400", icon: AlertCircle }
    };
    return statusMap[status] || statusMap["pending"];
};

export default function ProjectsPage() {
    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="font-display text-3xl font-black text-white text-glow-purple mb-2">Projects</h1>
                    <p className="text-slate-400 text-sm">Organize and manage your translation projects</p>
                </div>
                <button className="btn-border-beam group">
                    <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-2.5 px-5">
                        <Plus className="w-4 h-4" />
                        <span className="text-sm font-semibold">New Project</span>
                    </div>
                </button>
            </div>

            {/* Project Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {projects.map((project, index) => {
                    const statusConfig = getStatusBadge(project.status);
                    const StatusIcon = statusConfig.icon;

                    return (
                        <motion.div
                            key={project.id}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.1 }}
                            whileHover={{ y: -2 }}
                            className="glass-panel-glow p-5 cursor-pointer group"
                        >
                            <div className="glass-shine" />
                            <div className="relative z-10">
                                <div className="flex items-start justify-between mb-3">
                                    <div className="flex items-center gap-3">
                                        <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-primary-purple/10 border border-primary-purple/20">
                                            <Folder className="w-5 h-5 text-primary-purple-bright" />
                                        </div>
                                        <div>
                                            <h3 className="text-white font-bold text-base leading-tight">{project.name}</h3>
                                            <p className="text-slate-400 text-xs">{project.type}</p>
                                        </div>
                                    </div>
                                    <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full ${statusConfig.bg} border border-white/10`}>
                                        <StatusIcon className={`w-3 h-3 ${statusConfig.text}`} />
                                        <span className={`text-xs font-semibold capitalize ${statusConfig.text}`}>{project.status.replace('-', ' ')}</span>
                                    </div>
                                </div>

                                <div className="flex items-center gap-4 text-xs text-slate-500">
                                    <span>{project.files} file{project.files > 1 ? 's' : ''}</span>
                                    <span>•</span>
                                    <span>{project.date}</span>
                                </div>
                            </div>
                        </motion.div>
                    );
                })}
            </div>

            {/* Empty State for New Users */}
            {projects.length === 0 && (
                <div className="glass-panel p-12 text-center">
                    <Folder className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                    <h3 className="text-white text-lg font-bold mb-2">No projects yet</h3>
                    <p className="text-slate-400 text-sm mb-6">Create your first project to organize your translations</p>
                    <button className="btn-border-beam group">
                        <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-2.5 px-5">
                            <Plus className="w-4 h-4" />
                            <span className="text-sm font-semibold">Create Project</span>
                        </div>
                    </button>
                </div>
            )}
        </div>
    );
}

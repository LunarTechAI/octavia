"use client";

import { motion } from "framer-motion";

const languages = [
    "English", "Spanish", "French", "German", "Italian", "Portuguese",
    "Japanese", "Chinese", "Korean", "Russian", "Arabic", "Hindi",
    "Turkish", "Dutch", "Polish", "Swedish", "Indonesian", "Vietnamese"
];

export function GlobalScale() {
    return (
        <section className="py-24 relative overflow-hidden">
            <div className="container mx-auto px-6 text-center">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    className="mb-16"
                >
                    <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">
                        Speak their language.
                    </h2>
                    <p className="text-gray-400 max-w-2xl mx-auto">
                        Octavia supports over 30 languages with native-level fluency.
                        Expand your reach to billions of new viewers instantly.
                    </p>
                </motion.div>

                <div className="flex flex-wrap justify-center gap-3 max-w-4xl mx-auto">
                    {languages.map((lang, index) => (
                        <motion.div
                            key={lang}
                            initial={{ opacity: 0, scale: 0.9 }}
                            whileInView={{ opacity: 1, scale: 1 }}
                            viewport={{ once: true }}
                            transition={{ delay: index * 0.02 }}
                            className="px-4 py-2 rounded-full bg-white/5 border border-white/10 text-gray-300 hover:border-primary-purple/50 hover:bg-primary-purple/10 hover:text-white transition-all cursor-default"
                        >
                            {lang}
                        </motion.div>
                    ))}
                    <div className="px-4 py-2 rounded-full border border-dashed border-white/10 text-gray-500">
                        +12 more
                    </div>
                </div>
            </div>
        </section>
    );
}

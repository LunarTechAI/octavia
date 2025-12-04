
import { SubtitleNav } from "@/components/dashboard/SubtitleNav";

export default function SubtitlesLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col h-full">
      <SubtitleNav />
      <main className="flex-1 overflow-y-auto p-6">
        {children}
      </main>
    </div>
  );
}
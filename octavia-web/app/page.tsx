
import { Navbar } from "@/components/landing/Navbar";
import { HeroSection } from "@/components/landing/HeroSection";
import { PartnersSlider } from "@/components/landing/PartnersSlider";
import { FeaturesGrid } from "@/components/landing/FeaturesGrid";
import { HowItWorks } from "@/components/landing/HowItWorks";
import { LiveDemo } from "@/components/landing/LiveDemo";
import { GlobalScale } from "@/components/landing/GlobalScale";
import { PricingPreview } from "@/components/landing/PricingPreview";
import { CTAFooter } from "@/components/landing/CTAFooter";

// Main Landing Page
export default function Home() {
  return (
    <main className="min-h-screen selection:bg-primary-purple/30 selection:text-white">
      <Navbar />
      <HeroSection />
      <PartnersSlider />
      <FeaturesGrid />
      <HowItWorks />
      <LiveDemo />
      <GlobalScale />
      <PricingPreview />
      <CTAFooter />
    </main>
  );
}

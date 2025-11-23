# Dashboard Pages Map

This document maps the existing HTML prototypes in the `dashboards/` directory against the required pages for the Octavia Cloud Platform (SaaS).

## Page Inventory

| Category | Page Name | Status | Existing Path / Notes | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Core** | **Landing Page** | 🟢 Completed (2025-11-23) | `octavia-web/app/page.tsx` | Public-facing home page (Marketing + Login). |
| **Core** | **Hub / Dashboard** | 🟢 Existing | `dashboards/hub/ai_suite_hub.html` | Central hub to select tools (Video, Audio, Subtitles). |
| **Auth** | **Login / Sign Up** | 🔴 Missing | (Handled by Clerk) | Authentication pages (or Clerk components). |
| **Video** | **Video Translator Input** | 🟢 Existing | `dashboards/video-translation/video_ai_translator.html` | Upload video, select languages. |
| **Video** | **Translation Progress** | 🟢 Existing | `dashboards/video-translation/translation_in_progress.html` | Real-time progress bar and steps. |
| **Video** | **Review & Export** | 🟢 Existing | `dashboards/video-translation/video_translation_review.html` | Player, stats, and download options. |
| **Audio** | **Audio Translator** | 🟢 Existing | `dashboards/audio/audio_translation.html` | Input for audio-only translation. |
| **Audio** | **Subtitle to Audio** | 🟢 Existing | `dashboards/audio/subtitle_to_audio.html` | Convert SRT to spoken audio. |
| **Subtitles** | **Subtitle Gen Input** | 🟢 Existing | `dashboards/subtitles/subtitle_generation_input.html` | Generate subtitles from video/audio. |
| **Subtitles** | **Subtitle Gen Progress** | 🟢 Existing | `dashboards/subtitles/subtitle_generation_progress.html` | Progress for subtitle generation. |
| **Subtitles** | **Subtitle Review** | 🟢 Existing | `dashboards/subtitles/generated_subtitles_review_export.html` | Edit and export generated subtitles. |
| **Subtitles** | **Subtitle Translator** | 🟢 Existing | `dashboards/subtitles/subtitles_translation.html` | Translate existing SRT files. |
| **Settings** | **General Settings** | 🟢 Existing | `dashboards/settings/settings_preferences.html` | Basic app preferences. |
| **Settings** | **Advanced Settings** | 🟢 Existing | `dashboards/settings/settings_with_tabs.html` | Tabbed settings (likely place for Magic Mode toggles). |
| **Magic** | **My Voices** | 🔴 Missing | - | Manage cloned voices and samples. |
| **Billing** | **Plans & Billing** | 🔴 Missing | - | Subscription management, credit usage, invoices. |
| **Jobs** | **Job History** | 🟡 Partial | (Hub might cover this?) | List of past translations with status/download links. |
| **Account** | **Profile & Security** | 🔴 Missing | - | User profile, password change, MFA settings (Clerk). |
| **Account** | **Team / Organization** | 🔴 Missing | - | Manage team members and roles (Clerk). |

## Summary

- **Total Pages Required**: ~19
- **Existing Prototypes**: 12
- **Missing / To Build**: 7 (Landing, Auth, My Voices, Billing, Job History, Profile, Team)

## Next Steps

1.  **Migrate Existing**: Convert HTML prototypes to Next.js (React) components.
2.  **Build Missing**: Create new pages for "My Voices" and "Billing".
3.  **Integrate Auth**: Replace missing Auth pages with Clerk integration.

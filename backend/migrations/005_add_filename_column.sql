-- Migration to add filename column to translation_jobs table
-- This resolves the "Could not find the 'filename' column" error

-- Add the filename column if it doesn't exist
ALTER TABLE public.translation_jobs 
ADD COLUMN IF NOT EXISTS filename text;

-- Create index for better performance (optional)
CREATE INDEX IF NOT EXISTS idx_translation_jobs_filename 
ON public.translation_jobs USING btree (filename);

-- Add comment for documentation
COMMENT ON COLUMN public.translation_jobs.filename IS 'Generated output filename (e.g., subtitle_audio_xxx.mp3, translated_video_xxx.mp4)';

-- This migration ensures compatibility with code expecting filename column
-- The column stores the actual filename of generated files in the outputs directory

-- Migration to add download_url column to translation_jobs table
-- This resolves the "Could not find the 'download_url' column" error

-- Add the download_url column if it doesn't exist
ALTER TABLE public.translation_jobs 
ADD COLUMN IF NOT EXISTS download_url text;

-- Create index for better performance (optional)
CREATE INDEX IF NOT EXISTS idx_translation_jobs_download_url 
ON public.translation_jobs USING btree (download_url);

-- Add comment for documentation
COMMENT ON COLUMN public.translation_jobs.download_url IS 'Download URL for completed job results';

-- This migration ensures compatibility with code expecting download_url column
-- The column stores URLs like "/api/download/subtitle-audio/{job_id}" or "/api/download/video/{job_id}"

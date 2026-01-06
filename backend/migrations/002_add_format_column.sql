-- Add missing 'format' column to translation_jobs table
-- This fixes the PGRST204 error where the code expects 'format' but the table has 'output_format'

ALTER TABLE public.translation_jobs 
ADD COLUMN format text;

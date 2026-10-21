-- ==============================================================================
-- Fatty Liver AI Medical Detection System - Production Supabase Schema
-- Run this script in your Supabase Project SQL Editor:
-- https://supabase.com/dashboard/project/vazfzosxwqlakggkmwly/sql
-- ==============================================================================

-- 1. Enable UUID Extension (standard in PostgreSQL / Supabase)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ==============================================================================
-- 2. Create User Profiles Table (Synchronized with Supabase Auth)
-- ==============================================================================
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT,
    full_name TEXT,
    role TEXT DEFAULT 'Radiologist',
    institution TEXT DEFAULT 'Metropolitan Medical Center',
    avatar_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS for Profiles
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- Profiles Security Policies
CREATE POLICY "Public profiles are readable by authenticated users"
ON public.profiles FOR SELECT
TO authenticated
USING (true);

CREATE POLICY "Users can update their own profile"
ON public.profiles FOR UPDATE
TO authenticated
USING (auth.uid() = id);

CREATE POLICY "Users can insert their own profile"
ON public.profiles FOR INSERT
TO authenticated
WITH CHECK (auth.uid() = id);

-- Trigger to automatically populate profiles table upon auth.users signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (id, email, full_name, role)
  VALUES (
    NEW.id,
    NEW.email,
    COALESCE(NEW.raw_user_meta_data->>'full_name', split_part(NEW.email, '@', 1)),
    COALESCE(NEW.raw_user_meta_data->>'role', 'Radiologist')
  )
  ON CONFLICT (id) DO UPDATE
  SET email = EXCLUDED.email,
      full_name = EXCLUDED.full_name;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();


-- ==============================================================================
-- 3. Create Patient Ultrasound Predictions History Table
-- ==============================================================================
CREATE TABLE IF NOT EXISTS public.patient_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    record_id TEXT NOT NULL DEFAULT ('FL-' || to_hex(extract(epoch from now())::bigint)),
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    user_email TEXT,
    filename TEXT NOT NULL DEFAULT 'ultrasound_scan.jpg',
    prediction TEXT NOT NULL,
    confidence NUMERIC(5, 2) NOT NULL,
    severity TEXT NOT NULL,
    mode TEXT DEFAULT 'multiclass',
    probabilities JSONB DEFAULT '{}'::jsonb,
    acoustic_profile JSONB DEFAULT '{}'::jsonb,
    mean_echogenicity NUMERIC(6, 2),
    attenuation_slope NUMERIC(6, 3),
    processing_time NUMERIC(6, 3) DEFAULT 0.15,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for lightning-fast history filtering & sorting
CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON public.patient_predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON public.patient_predictions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_prediction ON public.patient_predictions(prediction);

-- Enable Row Level Security (RLS)
ALTER TABLE public.patient_predictions ENABLE ROW LEVEL SECURITY;

-- ==============================================================================
-- 4. Row Level Security Policies for Predictions
-- ==============================================================================

-- Authenticated Users: Read their own records or unassigned demo records
CREATE POLICY "Authenticated users can select their predictions"
ON public.patient_predictions
FOR SELECT
TO authenticated
USING (auth.uid() = user_id OR user_id IS NULL);

-- Authenticated Users: Insert new prediction records
CREATE POLICY "Authenticated users can insert predictions"
ON public.patient_predictions
FOR INSERT
TO authenticated
WITH CHECK (auth.uid() = user_id OR user_id IS NULL);

-- Authenticated Users: Delete their own prediction records
CREATE POLICY "Authenticated users can delete their predictions"
ON public.patient_predictions
FOR DELETE
TO authenticated
USING (auth.uid() = user_id);

-- Anonymous / Demo Access: Allow read and insert for immediate clinical testing
CREATE POLICY "Anon demo users can select predictions"
ON public.patient_predictions
FOR SELECT
TO anon
USING (true);

CREATE POLICY "Anon demo users can insert predictions"
ON public.patient_predictions
FOR INSERT
TO anon
WITH CHECK (true);

CREATE POLICY "Anon demo users can delete predictions"
ON public.patient_predictions
FOR DELETE
TO anon
USING (true);

-- ==============================================================================
-- 5. Helper Function: Auto-confirm User (Allows instant login without email SMTP)
-- ==============================================================================
CREATE OR REPLACE FUNCTION public.confirm_user_by_email(target_email TEXT)
RETURNS BOOLEAN AS $$
BEGIN
  UPDATE auth.users
  SET email_confirmed_at = NOW()
  WHERE email = target_email
    AND email_confirmed_at IS NULL;
  RETURN TRUE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant execute permissions to anon and service role
GRANT EXECUTE ON FUNCTION public.confirm_user_by_email(TEXT) TO anon, authenticated, service_role;

# 🚀 Production Deployment & Custom Domain Guide: `livercare.zayacodehub.in`

This document contains all copy-paste configurations, environment variables, DNS records, and Supabase settings required to deploy and connect your Fatty Liver Detection System.

---

## 📋 1. Supabase Cloud Configuration (Authentication & Database)

### A. URL Configuration (Authentication Redirects)
Navigate to **Supabase Dashboard** ➔ Select Project `vazfzosxwqlakggkmwly` ➔ **Authentication** ➔ **URL Configuration**:

- **Site URL**:
  ```text
  https://livercare.zayacodehub.in
  ```

- **Redirect URLs (Allow list)** *(Add each of these)*:
  ```text
  https://livercare.zayacodehub.in/**
  https://fatty-liver-gamma.vercel.app/**
  http://localhost:5000/**
  http://127.0.0.1:5000/**
  ```

---

### B. Supabase Database Schema (SQL Query Editor)
If not already created, run this in **Supabase Dashboard** ➔ **SQL Editor**:

```sql
-- 1. Create table for persistent ultrasound predictions
CREATE TABLE IF NOT EXISTS public.patient_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    patient_id TEXT NOT NULL,
    patient_name TEXT,
    age_gender TEXT,
    exam_date DATE DEFAULT CURRENT_DATE,
    clinical_notes TEXT,
    prediction TEXT NOT NULL,
    confidence NUMERIC(5,2) NOT NULL,
    mode TEXT DEFAULT 'multiclass',
    probabilities JSONB NOT NULL,
    acoustic_biomarkers JSONB,
    processing_time NUMERIC(6,3),
    image_url TEXT,
    clinician_name TEXT,
    hospital_name TEXT,
    status TEXT DEFAULT 'completed'
);

-- 2. Enable Row Level Security (RLS)
ALTER TABLE public.patient_predictions ENABLE ROW LEVEL SECURITY;

-- 3. Security policies
CREATE POLICY "Allow public read access"
ON public.patient_predictions FOR SELECT
TO anon, authenticated
USING (true);

CREATE POLICY "Allow authenticated users to insert"
ON public.patient_predictions FOR INSERT
TO anon, authenticated
WITH CHECK (true);

-- 4. Fast indexing
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON public.patient_predictions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_patient_id ON public.patient_predictions(patient_id);
```

---

## 🌐 2. DNS Settings (Domain Registrar: Cloudflare, GoDaddy, Hostinger, etc.)

Add this DNS record in your domain management panel:

### If your Frontend is on Vercel:
| Type | Name / Host | Target / Points To | Proxy Status (Cloudflare) | TTL |
| :--- | :--- | :--- | :--- | :--- |
| `CNAME` | `livercare` | `64c4ffcd12fe2524.vercel-dns-017.com` *(or `cname.vercel-dns.com`)* | DNS Only (Grey Cloud) | 3600 (1 Hour) |

*(Note: Once Vercel issues the SSL certificate, you can turn Cloudflare Proxy Orange Cloud back on with SSL mode set to **Full (Strict)**).*

### If connecting directly to Render:
| Type | Name / Host | Target / Points To | Proxy Status (Cloudflare) | TTL |
| :--- | :--- | :--- | :--- | :--- |
| `CNAME` | `livercare` | `fatty-liver-ramiyaa.onrender.com` | DNS Only (Grey Cloud) | 3600 (1 Hour) |

---

## 💻 3. Vercel Dashboard Settings (Frontend App)

1. Open **[Vercel Dashboard](https://vercel.com/dashboard)**.
2. Select your project: `fatty-liver-gamma`.
3. Go to **Settings** ➔ **Domains**.
4. Enter:
   ```text
   livercare.zayacodehub.in
   ```
5. Click **Add**.
6. Wait 1–3 minutes until the status shows:
   - ✅ `Valid Configuration`
   - ✅ `Certificate Issued`

---

## ⚙️ 4. Render Server Environment Variables (Backend Deploy Server)

Go to **[Render Dashboard](https://dashboard.render.com/)** ➔ Select Web Service `FATTY_LIVER_RAMIYAA` ➔ **Environment**:

Add or verify the following Environment Variables:

```

### Render Build & Start Commands (if not using Dockerfile):
- **Build Command**:
  ```bash
  pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
  ```
- **Start Command**:
  ```bash
  gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 app:app
  ```

---

## 🔒 5. CORS Verification (Pre-configured in code)

Your backend in [`app.py`](file:///c:/PROJECTS/FATTY%20LIVER%20DETECTION%20SYSTEM/FATTY_LIVER_RAMIYAA/app.py) already has wild-card CORS headers enabled:

```python
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response
```

All requests from `https://livercare.zayacodehub.in` to `https://fatty-liver-ramiyaa.onrender.com` are automatically authorized.

---

## 🩺 6. Live Verification Checklist

1. Open your browser and navigate to:
   ```text
   https://livercare.zayacodehub.in
   ```
2. Verify SSL certificate lock icon 🔒 is active.
3. Sign In or Sign Up using email/password.
4. Navigate to **Settings** ➔ Ensure API Base URL is set to:
   ```text
   https://fatty-liver-ramiyaa.onrender.com
   ```
5. Click **Ping Server** ➔ Status will show: `Online (Render Cloud)`.
6. Go to **Upload & Predict** ➔ Select sample scan or upload an image ➔ Click **Run AI Diagnostic Analysis**.
7. Click **Download A4 Medical Report** ➔ Verify the preview modal, click **Print / Save as PDF (A4)** or **Download HTML**.

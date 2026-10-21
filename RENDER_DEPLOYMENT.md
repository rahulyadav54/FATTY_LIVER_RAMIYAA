# 🚀 Deploying Fatty Liver AI Dashboard on Render & Vercel

## Overview: Render vs. Vercel for PyTorch AI Models

| Feature | Render (Recommended) | Vercel |
| :--- | :--- | :--- |
| **Supports PyTorch & ResNet-50** | ✅ **YES** (Supports Docker & full Python runtime) | ❌ **NO direct model hosting** (Hard 250MB size limit; PyTorch is >750MB) |
| **All-in-One Deployment** | ✅ **YES** (Hosts both UI + AI model on 1 URL) | ⚠️ **Frontend only** (Must connect to external backend like Render) |
| **Free Tier Available** | ✅ **YES** (Free Web Service / Docker) | ✅ **YES** (Free for frontend) |
| **Custom Domain & HTTPS** | ✅ **YES** (Free SSL included) | ✅ **YES** (Free SSL included) |

---

## 🎯 Option 1: 1-Click All-in-One Deployment on Render (Recommended)

Render will host both your **modern Medical AI Dashboard** and your **Siamese ResNet-50 PyTorch model** together on a single public URL (e.g., `https://fatty-liver-ai-dashboard.onrender.com`).

### Step 1: Commit and Push Your Code to GitHub
Open your terminal in this repository and run:
```bash
git add .
git commit -m "Add modern medical dashboard UI and Render deployment configuration"
git push origin main
```

### Step 2: Deploy on Render.com (3 Minutes)
1. Go to **[https://render.com](https://render.com)** and sign in (using your GitHub account).
2. Click the **"New +"** button in the top right and select **"Web Service"**.
3. Choose **"Build and deploy from a Git repository"** and click **Next**.
4. Select your repository: `rahulyadav54/FATTY_LIVER_RAMIYAA`.
5. Render will automatically detect the settings from `render.yaml` and `Dockerfile`:
   - **Name**: `fatty-liver-ai-dashboard`
   - **Region**: Oregon (US West) or Frankfurt (EU)
   - **Environment**: **Docker** *(Recommended — already preconfigured for CPU PyTorch to build fast and stay under 512MB RAM)*
   - **Instance Type**: **Free**
6. Click **"Deploy Web Service"**.
7. Wait 2–3 minutes for the build to complete. Once finished, Render will display your live public URL!

---

## 🌐 Option 2: Deploy Frontend on Vercel + Backend on Render

If you specifically want a `.vercel.app` domain for your frontend:

### Why Vercel Cannot Host the Model Directly
Vercel serverless functions strictly enforce an unzipped size limit of **250 MB**. PyTorch (`torch` + `torchvision`) is over 750 MB, and `best_model.pth` is 95 MB. Attempting to run PyTorch directly on Vercel will trigger:
`Error: The Serverless Function size exceeds the maximum limit of 250 MB.`

### How the Hybrid Architecture Works:
1. **Backend (AI Model + Flask)**: Deployed on **Render** (provides API endpoint `https://your-app.onrender.com/predict`).
2. **Frontend (Dashboard UI)**: Deployed on **Vercel** (`https://your-app.vercel.app`).
3. In `static/js/dashboard.js`, update the API URL to point to your Render backend:
   ```javascript
   const API_BASE = 'https://fatty-liver-ai-dashboard.onrender.com';
   ```
4. Push to GitHub and import the repository into **Vercel** as a static project.

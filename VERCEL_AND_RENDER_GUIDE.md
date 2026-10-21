# 🚀 Deploying Frontend on Vercel + Backend on Render

This guide explains how to deploy your **Fatty Liver Detection System** using a **decoupled architecture**:
- 🌐 **Frontend (Modern Medical Dashboard)**: Deployed on **Vercel** (`https://fatty-liver.vercel.app`)
- 🧠 **Backend (PyTorch ResNet-50 AI Engine)**: Deployed on **Render** (`https://fatty-liver-ai-dashboard.onrender.com`)

---

## 🏗️ Architecture Diagram

```
+-----------------------------------+             +--------------------------------------+
|       VERCEL (Frontend Host)      |             |         RENDER (Backend Host)        |
|                                   |             |                                      |
|  - HTML5 Medical Dashboard        |             |  - Flask REST API (app.py)           |
|  - Vanilla CSS3 Design System     |  CORS Fetch |  - Siamese Neural Network            |
|  - Chart.js Clinical Graphs       | ----------> |  - ResNet-50 Encoder                 |
|  - Ultrasound Sample Scans        |             |  - Acoustic Profiling Engine         |
|  - Fast Global CDN Delivery       |             |  - PyTorch Weights (best_model.pth)  |
|                                   |             |                                      |
|  URL: your-app.vercel.app         |             |  URL: your-app.onrender.com          |
+-----------------------------------+             +--------------------------------------+
```

---

## Part 1: Deploy Backend on Render (Do this FIRST)

Your Vercel frontend needs a live API URL to send ultrasound images to.

1. Go to **[https://dashboard.render.com](https://dashboard.render.com)** and sign in with GitHub.
2. Click **New +** ➔ **Web Service**.
3. Connect your repository: `rahulyadav54/FATTY_LIVER_RAMIYAA`.
4. Configure the service:
   - **Name**: `fatty-liver-ai-dashboard` *(or any name you prefer)*
   - **Environment**: **Docker**
   - **Region**: Oregon or Frankfurt
   - **Plan**: **Free**
5. Click **Create Web Service**.
6. Wait 2–3 minutes for the build to complete.
7. **Copy your live Render backend URL** (e.g. `https://fatty-liver-ai-dashboard.onrender.com`).

---

## Part 2: Deploy Frontend on Vercel

1. Go to **[https://vercel.com](https://vercel.com)** and log in with your GitHub account.
2. Click **Add New...** ➔ **Project**.
3. Find your repository **`FATTY_LIVER_RAMIYAA`** and click **Import**.
4. Configure Project settings:
   - **Framework Preset**: **Other**
   - **Root Directory**: `./` (leave default)
   - **Build & Development Settings**: Leave empty (static site requires no build step)
5. Under **Environment Variables**, you can optionally add:
   - Key: `BACKEND_API_URL`
   - Value: Your Render URL from Part 1 (e.g. `https://fatty-liver-ai-dashboard.onrender.com`)
6. Click **Deploy**.
7. In ~15 seconds, Vercel will give you your live URL (e.g. `https://fatty-liver-ramiyaa.vercel.app`)!

---

## ⚡ How CORS & API Communication Works

- In `app.py`, full **CORS (Cross-Origin Resource Sharing)** headers are pre-configured:
  ```python
  @app.after_request
  def add_cors_headers(response):
      response.headers['Access-Control-Allow-Origin'] = '*'
      response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
      response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
      return response
  ```
- In `static/js/dashboard.js`, the app dynamically detects whether it's running on localhost or Vercel, and routes all prediction calls (`/predict`, `/api/status`, `/api/model-info`) directly to your Render backend.

---

## 🔄 Changing Backend URL on Vercel Anytime
If your Render service URL has a different name, you can set it in your browser console or in localStorage:
```javascript
localStorage.setItem('fl_backend_url', 'https://your-custom-render-name.onrender.com');
location.reload();
```

/**
 * Supabase Authentication & Cloud Sync Client
 * Fatty Liver Detection System
 */

(function (window) {
  'use strict';

  // Local storage keys
  const STORAGE_KEYS = {
    URL: 'fl_supabase_url',
    KEY: 'fl_supabase_anon_key',
    SESSION: 'fl_auth_session',
    USER: 'fl_auth_user'
  };

  // Demo user data when Supabase is not yet configured or demo mode is used
  const DEMO_USER = {
    id: 'demo-clinician-001',
    email: 'dr.yadav@hospital.org',
    user_metadata: {
      full_name: 'Dr. Rahul Yadav',
      role: 'Chief Radiologist & AI Lead',
      institution: 'Metropolitan Medical Center'
    }
  };

  class SupabaseAuthManager {
    constructor() {
      this.client = null;
      this.currentUser = null;
      this.isDemoMode = false;
      this.initClient();
    }

    /**
     * Initialize Supabase client from localStorage or window config
     */
    initClient() {
      const url = localStorage.getItem(STORAGE_KEYS.URL) || window.SUPABASE_URL || '';
      const key = localStorage.getItem(STORAGE_KEYS.KEY) || window.SUPABASE_ANON_KEY || '';

      if (url && key && window.supabase && typeof window.supabase.createClient === 'function') {
        try {
          this.client = window.supabase.createClient(url, key);
          console.log('✅ Supabase client initialized successfully');
        } catch (err) {
          console.error('Failed to initialize Supabase client:', err);
          this.client = null;
        }
      } else {
        this.client = null;
      }

      // Check stored session
      const storedUser = localStorage.getItem(STORAGE_KEYS.USER);
      if (storedUser) {
        try {
          this.currentUser = JSON.parse(storedUser);
          this.isDemoMode = this.currentUser.id === DEMO_USER.id;
        } catch (e) {
          this.currentUser = null;
        }
      }
    }

    /**
     * Check if user is currently authenticated
     */
    isAuthenticated() {
      return this.currentUser !== null;
    }

    /**
     * Check if Supabase keys are configured
     */
    isConfigured() {
      const url = localStorage.getItem(STORAGE_KEYS.URL);
      const key = localStorage.getItem(STORAGE_KEYS.KEY);
      return Boolean(url && key);
    }

    /**
     * Save Supabase Project URL and Anon Key
     */
    saveCredentials(url, key) {
      if (!url || !key) {
        throw new Error('Both Supabase URL and Anon Key are required.');
      }
      const cleanUrl = url.trim().replace(/\/$/, '');
      const cleanKey = key.trim();

      localStorage.setItem(STORAGE_KEYS.URL, cleanUrl);
      localStorage.setItem(STORAGE_KEYS.KEY, cleanKey);
      this.initClient();
      return true;
    }

    /**
     * Test active Supabase connection
     */
    async testConnection(url, key) {
      try {
        if (!window.supabase || typeof window.supabase.createClient !== 'function') {
          throw new Error('Supabase SDK not loaded yet.');
        }
        const tempClient = window.supabase.createClient(url.trim(), key.trim());
        const { error } = await tempClient.auth.getSession();
        if (error) throw error;
        return { success: true, message: 'Connection established successfully!' };
      } catch (err) {
        return { success: false, message: err.message || 'Failed to connect to Supabase.' };
      }
    }

    /**
     * Sign In with Email and Password
     */
    async signIn(email, password) {
      // If client is configured, authenticate via Supabase Auth
      if (this.client) {
        const { data, error } = await this.client.auth.signInWithPassword({
          email: email.trim(),
          password: password
        });

        if (error) {
          throw error;
        }

        this.currentUser = data.user;
        this.isDemoMode = false;
        localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(data.user));
        if (data.session) {
          localStorage.setItem(STORAGE_KEYS.SESSION, JSON.stringify(data.session));
        }
        return { user: data.user, isDemo: false };
      }

      // If not configured, check if user requested demo credentials
      if (email.toLowerCase().includes('demo') || email.toLowerCase().includes('doctor') || email.toLowerCase().includes('admin')) {
        return this.signInDemo();
      }

      throw new Error('Supabase credentials not configured yet. Click "Configure Supabase" below or use "Quick Clinician Demo Access".');
    }

    /**
     * Sign Up / Register new account
     */
    async signUp(email, password, fullName, role = 'Radiologist') {
      if (!this.client) {
        throw new Error('Supabase not configured. Click "Configure Supabase" to add your Project URL and Anon Key.');
      }

      const { data, error } = await this.client.auth.signUp({
        email: email.trim(),
        password: password,
        options: {
          data: {
            full_name: fullName.trim(),
            role: role
          }
        }
      });

      if (error) {
        throw error;
      }

      this.currentUser = data.user;
      this.isDemoMode = false;
      localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(data.user));
      return { user: data.user, isDemo: false };
    }

    /**
     * Quick Demo Sign In
     */
    signInDemo() {
      this.currentUser = DEMO_USER;
      this.isDemoMode = true;
      localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(DEMO_USER));
      return { user: DEMO_USER, isDemo: true };
    }

    /**
     * Sign Out
     */
    async signOut() {
      if (this.client && !this.isDemoMode) {
        try {
          await this.client.auth.signOut();
        } catch (e) {
          console.warn('Supabase signOut error:', e);
        }
      }

      this.currentUser = null;
      this.isDemoMode = false;
      localStorage.removeItem(STORAGE_KEYS.USER);
      localStorage.removeItem(STORAGE_KEYS.SESSION);
    }

    /**
     * Get Current Active User
     */
    getUser() {
      return this.currentUser;
    }

    /**
     * Sync prediction to Supabase table (if configured and table exists)
     */
    async syncPrediction(predictionData) {
      if (!this.client || this.isDemoMode || !this.currentUser) {
        return false;
      }

      try {
        const { error } = await this.client
          .from('patient_predictions')
          .insert([
            {
              user_id: this.currentUser.id,
              prediction: predictionData.prediction,
              confidence: predictionData.confidence,
              severity: predictionData.severity,
              mode: predictionData.mode,
              mean_echogenicity: predictionData.acoustic_profile ? predictionData.acoustic_profile.mean_echogenicity : null,
              attenuation_slope: predictionData.acoustic_profile ? predictionData.acoustic_profile.attenuation_slope : null,
              created_at: new Date().toISOString()
            }
          ]);

        if (error) {
          console.info('Note: patient_predictions table not found or requires RLS setup in Supabase:', error.message);
          return false;
        }
        return true;
      } catch (err) {
        console.warn('Sync error:', err);
        return false;
      }
    }
  }

  // Expose global instance
  window.SupabaseAuth = new SupabaseAuthManager();

})(window);

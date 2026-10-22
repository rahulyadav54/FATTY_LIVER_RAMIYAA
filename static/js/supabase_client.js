/**
 * Supabase Authentication & Cloud Sync Client
 * Fatty Liver Detection System - Production Configuration
 */

(function (window) {
  'use strict';

  // Production Supabase Project Credentials
  const DEFAULT_SUPABASE_URL = 'https://vazfzosxwqlakggkmwly.supabase.co';
  const DEFAULT_SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZhemZ6b3N4d3FsYWtnZ2ttd2x5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3OTI2MDg5MzksImV4cCI6MjEwODE4NDkzOX0.Vw7F_-kK4TtiZMQwXo5_0u8c53WBz927nYqfGKoBme0';

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
     * Get backend API base url (Render or localhost)
     */
    getBackendUrl() {
      if (typeof window.getApiBaseUrl === 'function') {
        return window.getApiBaseUrl();
      }
      if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        return '';
      }
      return localStorage.getItem('fl_backend_url') || window.BACKEND_API_URL || 'https://fatty-liver-ramiyaa.onrender.com';
    }

    /**
     * Initialize Supabase client from localStorage or default credentials
     */
    initClient() {
      const url = localStorage.getItem(STORAGE_KEYS.URL) || window.SUPABASE_URL || DEFAULT_SUPABASE_URL;
      const key = localStorage.getItem(STORAGE_KEYS.KEY) || window.SUPABASE_ANON_KEY || DEFAULT_SUPABASE_ANON_KEY;

      if (url && key && window.supabase && typeof window.supabase.createClient === 'function') {
        try {
          this.client = window.supabase.createClient(url, key, {
            auth: {
              persistSession: true,
              autoRefreshToken: true
            }
          });
          console.log('✅ Supabase production client initialized successfully');
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
          if (this.currentUser && (this.currentUser.id === DEMO_USER.id || this.currentUser.id === 'demo-clinician-001')) {
            // Purge legacy demo session to enforce legitimate authentication
            this.currentUser = null;
            this.isDemoMode = false;
            localStorage.removeItem(STORAGE_KEYS.USER);
            localStorage.removeItem(STORAGE_KEYS.SESSION);
          } else {
            this.isDemoMode = false;
          }
        } catch (e) {
          this.currentUser = null;
          this.isDemoMode = false;
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
      const url = localStorage.getItem(STORAGE_KEYS.URL) || window.SUPABASE_URL || DEFAULT_SUPABASE_URL;
      const key = localStorage.getItem(STORAGE_KEYS.KEY) || window.SUPABASE_ANON_KEY || DEFAULT_SUPABASE_ANON_KEY;
      return Boolean(url && key);
    }

    /**
     * Get active configured URL and Anon Key
     */
    getCredentials() {
      return {
        url: localStorage.getItem(STORAGE_KEYS.URL) || DEFAULT_SUPABASE_URL,
        key: localStorage.getItem(STORAGE_KEYS.KEY) || DEFAULT_SUPABASE_ANON_KEY
      };
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
        return { success: true, message: 'Supabase connection verified successfully!' };
      } catch (err) {
        return { success: false, message: err.message || 'Failed to connect to Supabase.' };
      }
    }

    /**
     * Auto-confirm user via backend helper if needed
     */
    async requestAutoConfirm(email) {
      try {
        const backend = this.getBackendUrl();
        await fetch(`${backend}/api/auth/confirm-user`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email: email })
        });
      } catch (e) {
        console.warn('Auto-confirm attempt notice:', e);
      }
    }

    /**
     * Sign In with Email and Password
     */
    async signIn(email, password) {
      const cleanEmail = email.trim();

      // If client is configured, authenticate via Supabase Auth
      if (this.client) {
        let { data, error } = await this.client.auth.signInWithPassword({
          email: cleanEmail,
          password: password
        });

        // Only auto-confirm if Supabase explicitly complains that the email is not confirmed
        const errMsg = (error && error.message) ? error.message.toLowerCase() : '';
        const isUnconfirmed = error && (
          errMsg.includes('not confirmed') ||
          errMsg.includes('unconfirmed') ||
          error.code === 'email_not_confirmed'
        );

        if (isUnconfirmed) {
          console.info('Requesting auto-confirmation for unconfirmed clinician account...');
          await this.requestAutoConfirm(cleanEmail);
          const retry = await this.client.auth.signInWithPassword({
            email: cleanEmail,
            password: password
          });
          data = retry.data;
          error = retry.error;
        }

        if (error) {
          const rawMsg = error.message || '';
          if (rawMsg.toLowerCase().includes('invalid login credentials') ||
              rawMsg.toLowerCase().includes('invalid credentials') ||
              error.status === 400) {
            const userFriendlyError = new Error('Invalid email or password. Please verify your credentials and try again.');
            userFriendlyError.isAuthError = true;
            userFriendlyError.status = 400;
            throw userFriendlyError;
          }
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

      throw new Error('Supabase client is not initialized. Please configure credentials.');
    }

    /**
     * Sign Up / Register new clinician account
     */
    async signUp(email, password, fullName, role = 'Radiologist') {
      if (!this.client) {
        throw new Error('Supabase client not initialized.');
      }

      const cleanEmail = email.trim();
      const cleanName = fullName.trim();

      const { data, error } = await this.client.auth.signUp({
        email: cleanEmail,
        password: password,
        options: {
          data: {
            full_name: cleanName,
            role: role
          }
        }
      });

      if (error) {
        throw error;
      }

      // Auto-confirm user account so clinician can immediately use system
      await this.requestAutoConfirm(cleanEmail);

      // Attempt auto-login
      try {
        const loginRes = await this.client.auth.signInWithPassword({
          email: cleanEmail,
          password: password
        });
        if (loginRes.data && loginRes.data.user) {
          this.currentUser = loginRes.data.user;
          this.isDemoMode = false;
          localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(loginRes.data.user));
          if (loginRes.data.session) {
            localStorage.setItem(STORAGE_KEYS.SESSION, JSON.stringify(loginRes.data.session));
          }
          return { user: loginRes.data.user, isDemo: false };
        }
      } catch (loginErr) {
        console.warn('Auto-login notice after signup:', loginErr);
      }

      this.currentUser = data.user;
      this.isDemoMode = false;
      localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(data.user));
      return { user: data.user, isDemo: false };
    }

    /**
     * Quick Demo Sign In (Disabled)
     */
    signInDemo() {
      throw new Error('Demo login has been removed. Please log in or register with your clinician credentials.');
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
     * Fetch prediction history from Supabase cloud strictly for the active user
     */
    async fetchHistory(limit = 50) {
      if (!this.client) return [];

      // Strict per-user isolation: User must be authenticated to fetch their records
      if (!this.currentUser || !this.currentUser.id) {
        return [];
      }

      try {
        const { data, error } = await this.client
          .from('patient_predictions')
          .select('*')
          .eq('user_id', this.currentUser.id)
          .order('created_at', { ascending: false })
          .limit(limit);

        if (error) {
          console.info('Supabase cloud history notice:', error.message);
          return [];
        }

        return (data || []).map(record => ({
          id: record.record_id || record.id,
          timestamp: new Date(record.created_at).toLocaleString(),
          filename: record.filename || 'ultrasound_scan.jpg',
          mode: record.mode || 'multiclass',
          prediction: record.prediction,
          confidence: Number(record.confidence),
          severity: record.severity || (record.prediction === 'Normal' ? 'NORMAL' : 'ABNORMAL'),
          processing_time: record.processing_time || 0.15,
          isCloud: true,
          created_at: record.created_at
        }));
      } catch (err) {
        console.warn('Failed to fetch history from Supabase:', err);
        return [];
      }
    }

    /**
     * Sync prediction to Supabase table (patient_predictions) tied to active user
     */
    async syncPrediction(predictionData, file) {
      if (!this.client || !this.currentUser || !this.currentUser.id) {
        return false;
      }

      try {
        const userId = this.currentUser.id;
        const userEmail = this.currentUser.email || 'clinician@hospital.org';
        const filename = (file && file.name) ? file.name : (predictionData.filename || 'ultrasound_scan.jpg');
        const recordId = 'FL-' + Date.now().toString(36).toUpperCase();

        const insertPayload = {
          record_id: recordId,
          user_id: userId,
          user_email: userEmail,
          filename: filename,
          prediction: predictionData.prediction,
          confidence: Number(predictionData.confidence),
          severity: predictionData.severity || (predictionData.prediction === 'Normal' ? 'Normal' : 'Abnormal - Requires Medical Attention'),
          mode: predictionData.mode || 'multiclass',
          probabilities: predictionData.probabilities || {},
          acoustic_profile: predictionData.acoustic_profile || {},
          mean_echogenicity: predictionData.acoustic_profile ? predictionData.acoustic_profile.mean_echogenicity : null,
          attenuation_slope: predictionData.acoustic_profile ? predictionData.acoustic_profile.attenuation_slope : null,
          processing_time: predictionData.processing_time || 0.15,
          created_at: new Date().toISOString()
        };

        const { data, error } = await this.client
          .from('patient_predictions')
          .insert([insertPayload])
          .select();

        if (error) {
          console.info('Supabase cloud sync notice (run supabase_schema.sql if table is not yet created):', error.message);
          return false;
        }

        console.log('☁️ Prediction successfully synced to Supabase cloud for user:', userId, recordId);
        return data ? data[0] : true;
      } catch (err) {
        console.warn('Sync error:', err);
        return false;
      }
    }

    /**
     * Delete prediction from Supabase cloud (user-scoped)
     */
    async deletePrediction(recordId) {
      if (!this.client || !this.currentUser || !this.currentUser.id) return false;

      try {
        const { error } = await this.client
          .from('patient_predictions')
          .delete()
          .eq('user_id', this.currentUser.id)
          .or(`record_id.eq.${recordId},id.eq.${recordId}`);

        if (error) {
          console.warn('Delete cloud record notice:', error.message);
          return false;
        }
        return true;
      } catch (err) {
        console.warn('Delete cloud record error:', err);
        return false;
      }
    }

    /**
     * Clear all prediction records strictly for the active user
     */
    async clearUserPredictions() {
      if (!this.client || !this.currentUser || !this.currentUser.id) return false;

      try {
        const { error } = await this.client
          .from('patient_predictions')
          .delete()
          .eq('user_id', this.currentUser.id);

        if (error) {
          console.warn('Clear predictions cloud notice:', error.message);
          return false;
        }
        return true;
      } catch (err) {
        return false;
      }
    }
  }

  // Expose global instance
  window.SupabaseAuth = new SupabaseAuthManager();

})(window);

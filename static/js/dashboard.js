/**
 * Fatty Liver Detection System - Medical AI Dashboard Client
 * Complete interactive logic for upload, real-time prediction, history, and settings
 */

// Backend API Base URL Resolver (Supports Vercel frontend -> Render backend)
function getApiBaseUrl() {
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    return '';
  }
  const savedUrl = localStorage.getItem('fl_backend_url') || window.BACKEND_API_URL;
  if (savedUrl) {
    return savedUrl.replace(/\/$/, '');
  }
  // Connect to live Render backend
  return 'https://fatty-liver-ramiyaa.onrender.com';
}

// Application State
const state = {
  currentView: 'dashboard',
  selectedFile: null,
  activeSample: 'normal',
  classificationMode: 'multiclass',
  history: JSON.parse(localStorage.getItem('fl_prediction_history') || '[]'),
  theme: localStorage.getItem('fl_theme') || 'light',
  samples: {},
  modelInfo: null
};

// DOM Elements
const elements = {
  // Navigation
  navItems: document.querySelectorAll('.nav-item'),
  views: document.querySelectorAll('.view-section'),
  pageTitle: document.getElementById('current-page-title'),
  breadcrumbPage: document.getElementById('breadcrumb-page'),
  mobileMenuBtn: document.getElementById('mobileMenuBtn'),
  sidebar: document.getElementById('sidebar'),
  themeToggleBtn: document.getElementById('themeToggleBtn'),
  themeIcon: document.getElementById('themeIcon'),

  // Upload Area
  dropzone: document.getElementById('uploadDropzone'),
  fileInput: document.getElementById('fileInput'),
  browseBtn: document.getElementById('browseBtn'),
  previewPanel: document.getElementById('previewPanel'),
  previewImg: document.getElementById('previewImg'),
  fileName: document.getElementById('fileName'),
  fileDims: document.getElementById('fileDims'),
  fileSize: document.getElementById('fileSize'),
  analyzeBtn: document.getElementById('analyzeBtn'),
  clearBtn: document.getElementById('clearBtn'),
  modeTabs: document.querySelectorAll('.mode-tab-btn'),

  // Sample Image Area
  samplePills: document.querySelectorAll('.sample-pill'),
  sampleImg: document.getElementById('sampleImg'),
  sampleBadge: document.getElementById('sampleBadge'),
  sampleDesc: document.getElementById('sampleDesc'),
  useSampleBtn: document.getElementById('useSampleBtn'),

  // Results Section
  predictionBanner: document.getElementById('predictionSummaryBanner'),
  diagnosisClass: document.getElementById('diagnosisClass'),
  severityPill: document.getElementById('severityPill'),
  confidenceVal: document.getElementById('confidenceVal'),
  processingTime: document.getElementById('processingTime'),
  diagnosisDesc: document.getElementById('diagnosisDesc'),
  printReportBtn: document.getElementById('printReportBtn'),
  copyJsonBtn: document.getElementById('copyJsonBtn'),

  // Probability Cards
  probCards: {
    'Normal': document.getElementById('card-Normal'),
    'Grade-I': document.getElementById('card-Grade-I'),
    'Grade-II': document.getElementById('card-Grade-II'),
    'Grade-III': document.getElementById('card-Grade-III'),
    'CLD': document.getElementById('card-CLD')
  },

  // History Elements
  historyTableBody: document.getElementById('historyTableBody'),
  emptyHistoryState: document.getElementById('emptyHistoryState'),
  historySearch: document.getElementById('historySearch'),
  historyFilter: document.getElementById('historyFilter'),
  exportCsvBtn: document.getElementById('exportCsvBtn'),
  clearHistoryBtn: document.getElementById('clearHistoryBtn'),

  // Status & Loaders
  loadingOverlay: document.getElementById('loadingOverlay'),
  loadingText: document.getElementById('loadingText'),
  toast: document.getElementById('toast'),
  topbarStatusText: document.getElementById('topbarStatusText'),
  deviceBadgeText: document.getElementById('deviceBadgeText')
};

// ==========================================================================
// Initialization
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  setupNavigation();
  setupUploadInteractions();
  setupSampleInteractions();
  setupHistory();
  setupGraphTabs();
  fetchSystemStatus();
  fetchModelInfo();
  renderHistory();
  initBaselineAnalyticsCharts();
});

// ==========================================================================
// Theme Management
// ==========================================================================
function initTheme() {
  document.documentElement.setAttribute('data-theme', state.theme);
  updateThemeIcon();

  if (elements.themeToggleBtn) {
    elements.themeToggleBtn.addEventListener('click', () => {
      state.theme = state.theme === 'light' ? 'dark' : 'light';
      document.documentElement.setAttribute('data-theme', state.theme);
      localStorage.setItem('fl_theme', state.theme);
      updateThemeIcon();
      showToast(`Switched to ${state.theme} mode`);
    });
  }
}

function updateThemeIcon() {
  if (!elements.themeIcon) return;
  if (state.theme === 'dark') {
    elements.themeIcon.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="5"></circle>
        <line x1="12" y1="1" x2="12" y2="3"></line>
        <line x1="12" y1="21" x2="12" y2="23"></line>
        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
        <line x1="1" y1="12" x2="3" y2="12"></line>
        <line x1="21" y1="12" x2="23" y2="12"></line>
        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
      </svg>`;
  } else {
    elements.themeIcon.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
      </svg>`;
  }
}

// ==========================================================================
// Navigation & Views
// ==========================================================================
function setupNavigation() {
  elements.navItems.forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const targetView = item.getAttribute('data-view');
      switchView(targetView);

      // Close mobile sidebar if open
      if (elements.sidebar.classList.contains('open')) {
        elements.sidebar.classList.remove('open');
      }
    });
  });

  if (elements.mobileMenuBtn) {
    elements.mobileMenuBtn.addEventListener('click', () => {
      elements.sidebar.classList.toggle('open');
    });
  }

  // Quick help link in topbar
  const helpBtn = document.getElementById('helpBtn');
  if (helpBtn) {
    helpBtn.addEventListener('click', () => switchView('about'));
  }
}

function switchView(viewName) {
  state.currentView = viewName;

  // Update active sidebar item
  elements.navItems.forEach(item => {
    if (item.getAttribute('data-view') === viewName) {
      item.classList.add('active');
    } else {
      item.classList.remove('active');
    }
  });

  // Switch view DOM elements
  elements.views.forEach(view => {
    if (view.id === `${viewName}-view`) {
      view.classList.add('active');
    } else {
      view.classList.remove('active');
    }
  });

  // Update breadcrumb
  const titles = {
    'dashboard': 'Clinical Dashboard',
    'predict': 'Ultrasound Analysis',
    'history': 'Prediction History',
    'model': 'Model Specifications',
    'about': 'About the System',
    'settings': 'Preferences & Settings'
  };

  const title = titles[viewName] || 'Dashboard';
  if (elements.pageTitle) elements.pageTitle.textContent = title;
  if (elements.breadcrumbPage) elements.breadcrumbPage.textContent = title;

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ==========================================================================
// Upload & Dropzone Management
// ==========================================================================
function setupUploadInteractions() {
  if (!elements.dropzone) return;

  // Click triggers file input
  elements.dropzone.addEventListener('click', () => elements.fileInput.click());
  if (elements.browseBtn) {
    elements.browseBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      elements.fileInput.click();
    });
  }

  // Drag and drop events
  ['dragenter', 'dragover'].forEach(eventName => {
    elements.dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      elements.dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    elements.dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      elements.dropzone.classList.remove('dragover');
    });
  });

  elements.dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      processSelectedFile(files[0]);
    }
  });

  elements.fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      processSelectedFile(e.target.files[0]);
    }
  });

  // Clear / Reset
  if (elements.clearBtn) {
    elements.clearBtn.addEventListener('click', resetUploadState);
  }

  // Analyze Button
  if (elements.analyzeBtn) {
    elements.analyzeBtn.addEventListener('click', runInference);
  }

  // Classification Mode Toggle (5-class vs Binary)
  elements.modeTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      elements.modeTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      state.classificationMode = tab.getAttribute('data-mode');
      showToast(`Mode switched to: ${state.classificationMode === 'binary' ? 'Binary (Normal / Abnormal)' : '5-Class Multi-class'}`);
    });
  });

  // Action Buttons on Results
  if (elements.printReportBtn) {
    elements.printReportBtn.addEventListener('click', () => window.print());
  }

  if (elements.copyJsonBtn) {
    elements.copyJsonBtn.addEventListener('click', () => {
      if (state.lastResult) {
        navigator.clipboard.writeText(JSON.stringify(state.lastResult, null, 2));
        showToast('Result JSON copied to clipboard');
      }
    });
  }
}

function processSelectedFile(file) {
  // Validate format
  const validExtensions = ['jpg', 'jpeg', 'png', 'bmp'];
  const ext = file.name.split('.').pop().toLowerCase();
  if (!validExtensions.includes(ext)) {
    showToast('Invalid file format. Please upload JPG, PNG, or BMP.', 'error');
    return;
  }

  // Validate size (16MB max)
  if (file.size > 16 * 1024 * 1024) {
    showToast('File size exceeds 16MB limit.', 'error');
    return;
  }

  state.selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    elements.previewImg.src = e.target.result;
    
    // Get dimensions
    const imgObj = new Image();
    imgObj.onload = () => {
      if (elements.fileDims) {
        elements.fileDims.textContent = `${imgObj.naturalWidth} × ${imgObj.naturalHeight} px`;
      }
    };
    imgObj.src = e.target.result;

    if (elements.fileName) elements.fileName.textContent = file.name;
    if (elements.fileSize) elements.fileSize.textContent = `${(file.size / 1024).toFixed(1)} KB`;

    // Show preview, hide dropzone
    elements.dropzone.style.display = 'none';
    elements.previewPanel.style.display = 'flex';
  };
  reader.readAsDataURL(file);
}

function resetUploadState() {
  state.selectedFile = null;
  elements.fileInput.value = '';
  elements.previewPanel.style.display = 'none';
  elements.dropzone.style.display = 'block';
  elements.previewImg.src = '';
  if (elements.predictionBanner) {
    elements.predictionBanner.classList.remove('show');
  }
  resetProbabilityBars();
}

function resetProbabilityBars() {
  const classNames = ['Normal', 'Grade-I', 'Grade-II', 'Grade-III', 'CLD'];
  classNames.forEach(className => {
    const card = elements.probCards[className];
    if (card) {
      card.classList.remove('predicted-winner');
      const pctEl = card.querySelector('.prob-card-pct');
      const barFill = card.querySelector('.prob-bar-fill');
      if (pctEl) pctEl.textContent = '0.0%';
      if (barFill) barFill.style.width = '0%';
    }
  });
}

// ==========================================================================
// Sample Ultrasound Image Handling
// ==========================================================================
function setupSampleInteractions() {
  // Sample definitions with clinical descriptions
  state.samples = {
    'normal': {
      label: 'Normal (Healthy Liver)',
      url: '/static/images/samples/sample_normal.jpg',
      filename: 'sample_normal.jpg',
      badge: 'Normal Control Scan',
      desc: 'Homogeneous parenchyma, normal liver-to-kidney contrast, preserved portal vein wall echogenicity, and sharp diaphragm contour.'
    },
    'grade1': {
      label: 'Grade I (Mild Steatosis)',
      url: '/static/images/samples/sample_grade1.jpg',
      filename: 'sample_grade1.jpg',
      badge: 'Mild Steatosis (5-35%)',
      desc: 'Slight diffuse increase in hepatic parenchymal echogenicity with preserved diaphragm visualization and visible portal vein borders.'
    },
    'grade2': {
      label: 'Grade II (Moderate Steatosis)',
      url: '/static/images/samples/sample_grade2.jpg',
      filename: 'sample_grade2.jpg',
      badge: 'Moderate Steatosis (35-65%)',
      desc: 'Moderate diffuse echogenicity increase with acoustic attenuation in deeper parenchyma and impaired portal vessel border definition.'
    },
    'grade3': {
      label: 'Grade III (Severe Steatosis)',
      url: '/static/images/samples/sample_grade3.jpg',
      filename: 'sample_grade3.jpg',
      badge: 'Severe Steatosis (>65%)',
      desc: 'Severe diffuse hyperechogenicity with prominent posterior acoustic beam attenuation, poor penetration, and obscured diaphragm visualization.'
    },
    'cld': {
      label: 'CLD (Chronic Liver Disease)',
      url: '/static/images/samples/sample_cld.jpg',
      filename: 'sample_cld.jpg',
      badge: 'Fibrosis / Chronic Damage',
      desc: 'Coarse, heterogeneous liver echotexture with irregular contour, blunt margins, and architectural vascular distortion characteristic of fibrosis.'
    }
  };

  // Sample pill click
  elements.samplePills.forEach(pill => {
    pill.addEventListener('click', () => {
      elements.samplePills.forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      const sampleId = pill.getAttribute('data-sample');
      loadSampleDetails(sampleId);
    });
  });

  // "Use Sample Image" action
  if (elements.useSampleBtn) {
    elements.useSampleBtn.addEventListener('click', () => {
      const sample = state.samples[state.activeSample];
      if (!sample) return;

      showLoading(true, 'Loading sample ultrasound scan...');

      fetch(sample.url)
        .then(res => {
          if (!res.ok) throw new Error('Failed to load sample image');
          return res.blob();
        })
        .then(blob => {
          const file = new File([blob], sample.filename, { type: 'image/jpeg' });
          processSelectedFile(file);
          showLoading(false);
          showToast(`Loaded ${sample.label} into analysis panel`);
          
          // Switch to dashboard view if not already there
          if (state.currentView !== 'dashboard') {
            switchView('dashboard');
          }
        })
        .catch(err => {
          showLoading(false);
          showToast('Failed to load sample: ' + err.message, 'error');
        });
    });
  }
}

function loadSampleDetails(sampleId) {
  state.activeSample = sampleId;
  const sample = state.samples[sampleId];
  if (!sample) return;

  if (elements.sampleImg) elements.sampleImg.src = sample.url;
  if (elements.sampleBadge) elements.sampleBadge.textContent = sample.badge;
  if (elements.sampleDesc) elements.sampleDesc.textContent = sample.desc;
}

// ==========================================================================
// Inference Engine API Call (/predict)
// ==========================================================================
function runInference() {
  if (!state.selectedFile) {
    showToast('Please select or upload an ultrasound image first.', 'error');
    return;
  }

  const formData = new FormData();
  formData.append('file', state.selectedFile);
  formData.append('mode', state.classificationMode);

  showLoading(true, 'Analyzing ultrasound with Siamese ResNet-50...');
  elements.analyzeBtn.disabled = true;

  fetch(`${getApiBaseUrl()}/predict`, {
    method: 'POST',
    body: formData
  })
    .then(async (response) => {
      const data = await response.json();
      if (!response.ok || data.error) {
        throw new Error(data.error || 'Server returned an inference error');
      }
      return data;
    })
    .then(result => {
      showLoading(false);
      elements.analyzeBtn.disabled = false;
      state.lastResult = result;
      displayPredictionResults(result);
      saveToHistory(result, state.selectedFile);
      showToast(`Inference complete: ${result.prediction}`);
    })
    .catch(err => {
      showLoading(false);
      elements.analyzeBtn.disabled = false;
      showToast(err.message, 'error');
    });
}

function displayPredictionResults(data) {
  // 1. Show Prediction Summary Banner
  if (elements.predictionBanner) {
    elements.predictionBanner.classList.add('show');
  }

  // Diagnosis label & styling
  const winnerClass = data.top_candidate || data.prediction;
  if (elements.diagnosisClass) {
    elements.diagnosisClass.textContent = data.prediction;
    
    // Class-specific styling
    const classColorMap = {
      'Normal': 'var(--color-normal-bg)',
      'Grade-I': 'var(--color-grade1-bg)',
      'Grade-II': 'var(--color-grade2-bg)',
      'Grade-III': 'var(--color-grade3-bg)',
      'CLD': 'var(--color-cld-bg)',
      'Abnormal': 'var(--color-grade3-bg)'
    };
    
    const textColorMap = {
      'Normal': 'var(--color-normal)',
      'Grade-I': 'var(--color-grade1)',
      'Grade-II': 'var(--color-grade2)',
      'Grade-III': 'var(--color-grade3)',
      'CLD': 'var(--color-cld)',
      'Abnormal': 'var(--color-grade3)'
    };

    elements.diagnosisClass.style.backgroundColor = classColorMap[winnerClass] || 'var(--primary-light)';
    elements.diagnosisClass.style.color = textColorMap[winnerClass] || 'var(--primary)';
  }

  // Severity indicator
  if (elements.severityPill) {
    const isNormal = (winnerClass === 'Normal');
    const isUncertain = (data.status === 'uncertain');
    elements.severityPill.textContent = isUncertain ? 'UNCERTAIN — FURTHER REVIEW' : (isNormal ? 'NORMAL' : 'ABNORMAL — REQUIRES ATTENTION');
    elements.severityPill.className = `severity-pill ${isNormal && !isUncertain ? 'normal' : 'abnormal'}`;
  }

  // Confidence & Processing Time
  if (elements.confidenceVal) {
    elements.confidenceVal.textContent = `${data.confidence}%`;
  }
  if (elements.processingTime) {
    elements.processingTime.textContent = `${data.processing_time || 0.12}s`;
  }
  if (elements.diagnosisDesc) {
    elements.diagnosisDesc.textContent = data.message || data.description || 'Ultrasound steatosis classification complete.';
  }

  // 2. Animate Probability Cards
  resetProbabilityBars();

  if (data.probabilities) {
    for (const [className, prob] of Object.entries(data.probabilities)) {
      const card = elements.probCards[className];
      if (card) {
        const pctEl = card.querySelector('.prob-card-pct');
        const barFill = card.querySelector('.prob-bar-fill');
        
        if (pctEl) pctEl.textContent = `${prob.toFixed(1)}%`;
        if (barFill) barFill.style.width = `${prob}%`;

        // Highlight predicted winner
        if (className === winnerClass) {
          card.classList.add('predicted-winner');
        }
      }
    }
  }

  // Scroll smoothly down to results
  elements.predictionBanner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  // Render Interactive Clinical Graphs
  updateAnalyticsCharts(data);
}

// ==========================================================================
// Ultrasound Analytics & Interactive Charts (Chart.js)
// ==========================================================================
let charts = {
  probabilities: null,
  attenuation: null,
  histogram: null,
  radar: null
};

function setupGraphTabs() {
  const tabs = document.querySelectorAll('.graph-tab-btn');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const targetGraph = tab.getAttribute('data-graph');

      document.querySelectorAll('.chart-container').forEach(c => c.classList.remove('active'));
      const activeContainer = document.getElementById(`chart-${targetGraph}-wrap`);
      if (activeContainer) {
        activeContainer.classList.add('active');
        if (charts[targetGraph]) {
          charts[targetGraph].resize();
          charts[targetGraph].update();
        }
      }
    });
  });
}

function updateAnalyticsCharts(data) {
  if (typeof Chart === 'undefined') {
    console.warn('Chart.js not loaded yet');
    return;
  }

  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  const textColor = isDark ? '#94a3b8' : '#64748b';
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.06)';

  // 1. Update Metric Chips
  const profile = data.acoustic_profile;
  if (profile) {
    const elEcho = document.getElementById('valMeanEcho');
    const elSlope = document.getElementById('valAttenuationSlope');
    const elSpeckle = document.getElementById('valSpeckle');
    const elStatus = document.getElementById('valDiagnosticStatus');

    if (elEcho) elEcho.textContent = `${profile.mean_echogenicity} / 255`;
    if (elSlope) elSlope.textContent = `${profile.attenuation_slope} dB/cm`;
    if (elSpeckle) elSpeckle.textContent = `±${profile.speckle_variation}`;
    if (elStatus) {
      elStatus.textContent = data.prediction;
      elStatus.style.color = (data.prediction === 'Normal' ? 'var(--color-normal)' : 'var(--color-grade2)');
    }
  }

  // 2. Chart 1: Probability Distribution Bar Chart
  const ctxProb = document.getElementById('chartProbabilities');
  if (ctxProb) {
    if (charts.probabilities) charts.probabilities.destroy();

    const probLabels = Object.keys(data.probabilities || {});
    const probValues = Object.values(data.probabilities || {});
    const colorMap = {
      'Normal': '#10b981',
      'Grade-I': '#2563eb',
      'Grade-II': '#f59e0b',
      'Grade-III': '#ef4444',
      'CLD': '#8b5cf6',
      'Abnormal': '#ef4444'
    };
    const barColors = probLabels.map(l => colorMap[l] || '#2478f3');

    charts.probabilities = new Chart(ctxProb, {
      type: 'bar',
      data: {
        labels: probLabels,
        datasets: [{
          label: 'Prediction Probability (%)',
          data: probValues,
          backgroundColor: barColors,
          borderRadius: 8,
          borderWidth: 0,
          barThickness: 28
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => ` Probability: ${ctx.parsed.x.toFixed(1)}%`
            }
          }
        },
        scales: {
          x: {
            min: 0,
            max: 100,
            ticks: { color: textColor, callback: (v) => v + '%' },
            grid: { color: gridColor }
          },
          y: {
            ticks: { color: textColor, font: { weight: '600' } },
            grid: { display: false }
          }
        }
      }
    });
  }

  // 3. Chart 2: Acoustic Depth Attenuation Profile
  const ctxAtten = document.getElementById('chartAttenuation');
  if (ctxAtten && profile) {
    if (charts.attenuation) charts.attenuation.destroy();

    charts.attenuation = new Chart(ctxAtten, {
      type: 'line',
      data: {
        labels: profile.depth_labels,
        datasets: [
          {
            label: 'Patient Parenchyma Sound Transmission',
            data: profile.depth_profile,
            borderColor: '#2478f3',
            backgroundColor: 'rgba(36, 120, 243, 0.12)',
            fill: true,
            tension: 0.35,
            pointBackgroundColor: '#2478f3',
            pointRadius: 4
          },
          {
            label: 'Healthy Liver Reference Baseline',
            data: profile.normal_reference,
            borderColor: '#10b981',
            borderDash: [5, 5],
            fill: false,
            tension: 0.2,
            pointBackgroundColor: '#10b981',
            pointRadius: 3
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: { color: textColor, font: { size: 12, weight: '600' } }
          },
          tooltip: { mode: 'index', intersect: false }
        },
        scales: {
          x: {
            ticks: { color: textColor },
            grid: { color: gridColor }
          },
          y: {
            ticks: { color: textColor },
            grid: { color: gridColor },
            title: { display: true, text: 'Acoustic Signal Amplitude', color: textColor }
          }
        }
      }
    });
  }

  // 4. Chart 3: Echogenicity Gray-level Histogram
  const ctxHist = document.getElementById('chartHistogram');
  if (ctxHist && profile) {
    if (charts.histogram) charts.histogram.destroy();

    charts.histogram = new Chart(ctxHist, {
      type: 'bar',
      data: {
        labels: profile.histogram_bins,
        datasets: [{
          label: 'Tissue Echo Intensity Frequency (%)',
          data: profile.histogram_values,
          backgroundColor: 'rgba(36, 120, 243, 0.7)',
          hoverBackgroundColor: '#2478f3',
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => ` Frequency: ${ctx.parsed.y.toFixed(1)}%`
            }
          }
        },
        scales: {
          x: {
            ticks: { color: textColor, maxRotation: 45, font: { size: 10 } },
            grid: { display: false }
          },
          y: {
            ticks: { color: textColor, callback: (v) => v + '%' },
            grid: { color: gridColor }
          }
        }
      }
    });
  }

  // 5. Chart 4: Clinical Steatosis Radar
  const ctxRadar = document.getElementById('chartRadar');
  if (ctxRadar && profile) {
    if (charts.radar) charts.radar.destroy();

    // Normalize metrics into 0-100 radar scores
    const echoScore = Math.min(100, (profile.mean_echogenicity / 100) * 80);
    const attenScore = Math.min(100, Math.abs(profile.attenuation_slope) * 85);
    const textureScore = Math.min(100, (profile.speckle_variation / 80) * 85);
    const vesselBlurScore = data.prediction === 'Normal' ? 15 : (data.prediction === 'Grade-I' ? 45 : 85);
    const deepAttenuation = data.prediction === 'Normal' ? 10 : (data.prediction === 'Grade-I' ? 40 : (data.prediction === 'Grade-II' ? 75 : 95));

    charts.radar = new Chart(ctxRadar, {
      type: 'radar',
      data: {
        labels: [
          'Hyperechogenicity',
          'Beam Attenuation',
          'Parenchymal Noise',
          'Vascular Blurring',
          'Deep Penetration Loss'
        ],
        datasets: [
          {
            label: 'Detected Scan Indicators',
            data: [echoScore, attenScore, textureScore, vesselBlurScore, deepAttenuation],
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245, 158, 11, 0.2)',
            pointBackgroundColor: '#f59e0b',
            pointRadius: 4
          },
          {
            label: 'Normal Liver Threshold',
            data: [25, 20, 25, 20, 20],
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            borderDash: [4, 4],
            pointBackgroundColor: '#10b981',
            pointRadius: 3
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: textColor, font: { weight: '600' } }
          }
        },
        scales: {
          r: {
            angleLines: { color: gridColor },
            grid: { color: gridColor },
            pointLabels: { color: textColor, font: { size: 11, weight: '600' } },
            min: 0,
            max: 100,
            ticks: { display: false }
          }
        }
      }
  }
}

function initBaselineAnalyticsCharts() {
  const baselineData = {
    prediction: 'Normal',
    probabilities: {
      'Normal': 91.1,
      'Grade-I': 4.8,
      'Grade-II': 2.1,
      'Grade-III': 1.2,
      'CLD': 0.8
    },
    acoustic_profile: {
      mean_echogenicity: 88.5,
      attenuation_slope: -0.31,
      speckle_variation: 26.4,
      depth_labels: ['1.0 cm', '2.0 cm', '3.0 cm', '4.0 cm', '5.0 cm', '6.0 cm', '7.0 cm', '8.0 cm', '9.0 cm', '10.0 cm'],
      depth_profile: [142.5, 137.8, 130.6, 121.9, 113.8, 104.9, 95.8, 87.7, 79.4, 72.1],
      normal_reference: [145.0, 140.0, 132.0, 124.0, 115.0, 106.0, 97.0, 88.0, 80.0, 72.0],
      histogram_bins: ['0-15', '16-31', '32-47', '48-63', '64-79', '80-95', '96-111', '112-127', '128-143', '144-159', '160-175', '176-191', '192-207', '208-223', '224-239', '240-255'],
      histogram_values: [0.2, 0.8, 2.4, 5.1, 10.2, 16.8, 22.4, 19.1, 12.3, 6.2, 3.1, 1.0, 0.3, 0.1, 0.0, 0.0]
    }
  };
  updateAnalyticsCharts(baselineData);
}

// ==========================================================================
// Prediction History Management
// ==========================================================================
function setupHistory() {
  if (elements.historySearch) {
    elements.historySearch.addEventListener('input', renderHistory);
  }
  if (elements.historyFilter) {
    elements.historyFilter.addEventListener('change', renderHistory);
  }
  if (elements.clearHistoryBtn) {
    elements.clearHistoryBtn.addEventListener('click', () => {
      if (confirm('Are you sure you want to clear all prediction history records?')) {
        state.history = [];
        localStorage.removeItem('fl_prediction_history');
        renderHistory();
        showToast('Prediction history cleared');
      }
    });
  }
  if (elements.exportCsvBtn) {
    elements.exportCsvBtn.addEventListener('click', exportHistoryToCsv);
  }
}

function saveToHistory(result, file) {
  const record = {
    id: 'FL-' + Date.now().toString(36).toUpperCase(),
    timestamp: new Date().toLocaleString(),
    filename: file.name,
    mode: result.mode || 'multiclass',
    prediction: result.prediction,
    confidence: result.confidence,
    severity: result.severity,
    processing_time: result.processing_time
  };

  state.history.unshift(record);
  if (state.history.length > 50) state.history.pop(); // keep last 50
  localStorage.setItem('fl_prediction_history', JSON.stringify(state.history));
  renderHistory();
}

function renderHistory() {
  if (!elements.historyTableBody) return;

  const searchQuery = elements.historySearch ? elements.historySearch.value.toLowerCase() : '';
  const filterClass = elements.historyFilter ? elements.historyFilter.value : 'all';

  const filtered = state.history.filter(item => {
    const matchesSearch = item.filename.toLowerCase().includes(searchQuery) ||
                          item.prediction.toLowerCase().includes(searchQuery) ||
                          item.id.toLowerCase().includes(searchQuery);
    const matchesClass = filterClass === 'all' || item.prediction === filterClass;
    return matchesSearch && matchesClass;
  });

  if (filtered.length === 0) {
    elements.historyTableBody.innerHTML = '';
    elements.emptyHistoryState.style.display = 'block';
    return;
  }

  elements.emptyHistoryState.style.display = 'none';
  elements.historyTableBody.innerHTML = filtered.map(item => `
    <tr>
      <td><strong>${item.id}</strong></td>
      <td>${item.timestamp}</td>
      <td>${item.filename}</td>
      <td><span class="hero-tag">${item.mode}</span></td>
      <td>
        <span class="diagnosis-class-badge" style="font-size: 0.8rem; padding: 3px 8px;">
          ${item.prediction}
        </span>
      </td>
      <td><strong>${item.confidence}%</strong></td>
      <td>
        <span class="severity-pill ${item.prediction === 'Normal' ? 'normal' : 'abnormal'}" style="font-size: 0.7rem;">
          ${item.prediction === 'Normal' ? 'NORMAL' : 'ABNORMAL'}
        </span>
      </td>
      <td>
        <button class="btn-small btn-secondary" onclick="deleteHistoryItem('${item.id}')" title="Delete record">
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
        </button>
      </td>
    </tr>
  `).join('');
}

window.deleteHistoryItem = function(id) {
  state.history = state.history.filter(item => item.id !== id);
  localStorage.setItem('fl_prediction_history', JSON.stringify(state.history));
  renderHistory();
  showToast('Record deleted');
};

function exportHistoryToCsv() {
  if (state.history.length === 0) {
    showToast('No history records to export.', 'error');
    return;
  }

  const headers = ['Record ID', 'Timestamp', 'Filename', 'Mode', 'Predicted Class', 'Confidence (%)', 'Severity', 'Processing Time (s)'];
  const rows = state.history.map(item => [
    item.id,
    `"${item.timestamp}"`,
    `"${item.filename}"`,
    item.mode,
    item.prediction,
    item.confidence,
    `"${item.severity}"`,
    item.processing_time
  ]);

  const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', `fatty_liver_predictions_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  showToast('History exported to CSV file');
}

// ==========================================================================
// Status & Metadata Fetching
// ==========================================================================
function fetchSystemStatus() {
  fetch(`${getApiBaseUrl()}/api/status`)
    .then(res => res.json())
    .then(data => {
      if (elements.topbarStatusText) {
        elements.topbarStatusText.textContent = `Siamese ResNet-50 • ${data.device.toUpperCase()}`;
      }
      if (elements.deviceBadgeText) {
        elements.deviceBadgeText.textContent = `Device: ${data.device.toUpperCase()}`;
      }
    })
    .catch(() => {
      if (elements.topbarStatusText) {
        elements.topbarStatusText.textContent = 'Server Offline';
      }
    });
}

function fetchModelInfo() {
  fetch(`${getApiBaseUrl()}/api/model-info`)
    .then(res => res.json())
    .then(info => {
      state.modelInfo = info;
      populateModelSpecs(info);
    })
    .catch(err => console.error('Failed to load model info:', err));
}

function populateModelSpecs(info) {
  const container = document.getElementById('modelSpecsDynamic');
  if (!container || !info) return;

  container.innerHTML = `
    <div class="spec-row">
      <span class="spec-label">Model Architecture</span>
      <span class="spec-value">${info.model_name}</span>
    </div>
    <div class="spec-row">
      <span class="spec-label">Feature Encoder</span>
      <span class="spec-value">${info.architecture.encoder}</span>
    </div>
    <div class="spec-row">
      <span class="spec-label">Projection Head</span>
      <span class="spec-value">${info.architecture.projection_head}</span>
    </div>
    <div class="spec-row">
      <span class="spec-label">Classifier Head</span>
      <span class="spec-value">${info.architecture.classification_head}</span>
    </div>
    <div class="spec-row">
      <span class="spec-label">Input Resolution</span>
      <span class="spec-value">${info.architecture.input_dimensions}</span>
    </div>
    <div class="spec-row">
      <span class="spec-label">Target Multi-class Acc</span>
      <span class="spec-value" style="color: var(--color-normal);">${info.evaluation_framework.target_multiclass_accuracy}</span>
    </div>
    <div class="spec-row">
      <span class="spec-label">Target Binary Acc</span>
      <span class="spec-value" style="color: var(--color-normal);">${info.evaluation_framework.target_binary_accuracy}</span>
    </div>
    <div class="spec-row">
      <span class="spec-label">Multi-class ROC-AUC</span>
      <span class="spec-value" style="color: var(--primary);">${info.evaluation_framework.roc_auc_multiclass}</span>
    </div>
  `;
}

// ==========================================================================
// Utilities: Toast & Loader
// ==========================================================================
function showLoading(show, message = 'Processing...') {
  if (!elements.loadingOverlay) return;
  elements.loadingOverlay.style.display = show ? 'flex' : 'none';
  if (elements.loadingText) elements.loadingText.textContent = message;
}

function showToast(message, type = 'info') {
  if (!elements.toast) return;
  elements.toast.textContent = message;
  elements.toast.className = `toast show ${type === 'error' ? 'error' : ''}`;
  setTimeout(() => {
    elements.toast.classList.remove('show');
  }, 3500);
}

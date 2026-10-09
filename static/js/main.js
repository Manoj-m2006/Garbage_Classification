/**
 * EcoSort — Professional Platform & Interactive Bin Station Controller
 */

document.addEventListener('DOMContentLoaded', () => {
  const tabUpload = document.getElementById('tabUpload');
  const tabCamera = document.getElementById('tabCamera');
  const fileDropBox = document.getElementById('fileDropBox');
  const cameraContainer = document.getElementById('cameraContainer');
  const webcamVideo = document.getElementById('webcamVideo');
  const fileInput = document.getElementById('fileInput');
  const previewDisplayBox = document.getElementById('previewDisplayBox');
  const imagePreview = document.getElementById('imagePreview');
  const btnClearPreview = document.getElementById('btnClearPreview');
  const btnStartScan = document.getElementById('btnStartScan');

  const outPlaceholder = document.getElementById('outPlaceholder');
  const outLoading = document.getElementById('outLoading');
  const outResults = document.getElementById('outResults');

  const resCardBanner = document.getElementById('resCardBanner');
  const resBigEmoji = document.getElementById('resBigEmoji');
  const resClsName = document.getElementById('resClsName');
  const resBinTagPill = document.getElementById('resBinTagPill');
  const resScoreDigits = document.getElementById('resScoreDigits');
  const adviceTitleText = document.getElementById('adviceTitleText');
  const adviceText = document.getElementById('adviceText');
  const btnCopyAdvice = document.getElementById('btnCopyAdvice');
  const probBarsFlex = document.getElementById('probBarsFlex');

  const illustratedBins = document.querySelectorAll('.illustrated-bin');
  const thumbCards = document.querySelectorAll('.thumb-card');

  // Impact Calculator Elements
  const inputItemCount = document.getElementById('inputItemCount');
  const valCo2 = document.getElementById('valCo2');
  const valKwh = document.getElementById('valKwh');
  const valLandfill = document.getElementById('valLandfill');

  // FAQ Accordion Cards
  const faqItemCards = document.querySelectorAll('.faq-item-card');

  let selectedFile = null;
  let webcamStream = null;
  let activeTabMode = 'upload';

  const BIN_MAPPING = {
    'cardboard': { binId: 'binPurple', emoji: '📦', name: 'Paperboard Bin', cls: 'cardboard' },
    'paper':     { binId: 'binPurple', emoji: '📄', name: 'Paperboard Bin', cls: 'paper' },
    'glass':     { binId: 'binMint',   emoji: '🍾', name: 'Glass & Plastic Bin', cls: 'glass' },
    'plastic':   { binId: 'binMint',   emoji: '🥤', name: 'Glass & Plastic Bin', cls: 'plastic' },
    'metal':     { binId: 'binBrown',  emoji: '🥫', name: 'Metal & E-Waste Bin', cls: 'metal' },
    'trash':     { binId: 'binDark',   emoji: '🗑️', name: 'General Waste Bin', cls: 'trash' }
  };

  // Tab Switching
  tabUpload.addEventListener('click', () => switchTab('upload'));
  tabCamera.addEventListener('click', () => switchTab('camera'));

  function switchTab(mode) {
    activeTabMode = mode;
    if (mode === 'upload') {
      tabUpload.classList.add('active');
      tabCamera.classList.remove('active');
      fileDropBox.style.display = selectedFile ? 'none' : 'block';
      cameraContainer.style.display = 'none';
      stopWebcam();
    } else {
      tabCamera.classList.add('active');
      tabUpload.classList.remove('active');
      fileDropBox.style.display = 'none';
      previewDisplayBox.style.display = 'none';
      cameraContainer.style.display = 'block';
      startWebcam();
      btnStartScan.disabled = false;
    }
  }

  async function startWebcam() {
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } }
      });
      webcamVideo.srcObject = webcamStream;
    } catch (err) {
      alert('Unable to access camera.');
      switchTab('upload');
    }
  }

  function stopWebcam() {
    if (webcamStream) {
      webcamStream.getTracks().forEach(t => t.stop());
      webcamStream = null;
    }
  }

  // File Drag & Drop
  ['dragenter', 'dragover'].forEach(evt => {
    fileDropBox.addEventListener(evt, (e) => {
      e.preventDefault();
      fileDropBox.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(evt => {
    fileDropBox.addEventListener(evt, (e) => {
      e.preventDefault();
      fileDropBox.classList.remove('dragover');
    });
  });

  fileDropBox.addEventListener('drop', (e) => {
    if (e.dataTransfer.files.length > 0) loadFile(e.dataTransfer.files[0]);
  });

  fileDropBox.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) loadFile(e.target.files[0]);
  });

  btnClearPreview.addEventListener('click', (e) => {
    e.stopPropagation();
    resetState();
  });

  function loadFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please upload a valid image file (.jpg, .png, .webp).');
      return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      previewDisplayBox.style.display = 'block';
      fileDropBox.style.display = 'none';
      btnStartScan.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  function resetState() {
    selectedFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    previewDisplayBox.style.display = 'none';
    if (activeTabMode === 'upload') {
      fileDropBox.style.display = 'block';
      btnStartScan.disabled = true;
    }
    outPlaceholder.style.display = 'block';
    outLoading.style.display = 'none';
    outResults.style.display = 'none';
    closeAllBins();
  }

  // Scan Action
  btnStartScan.addEventListener('click', async () => {
    if (activeTabMode === 'camera') {
      const canvas = document.createElement('canvas');
      canvas.width = webcamVideo.videoWidth || 640;
      canvas.height = webcamVideo.videoHeight || 480;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(async (blob) => {
        if (!blob) return;
        const captured = new File([blob], 'camera_capture.jpg', { type: 'image/jpeg' });
        imagePreview.src = canvas.toDataURL('image/jpeg');
        previewDisplayBox.style.display = 'block';
        cameraContainer.style.display = 'none';
        stopWebcam();
        selectedFile = captured;
        await runClassification(captured);
      }, 'image/jpeg');
    } else if (selectedFile) {
      await runClassification(selectedFile);
    }
  });

  async function runClassification(file) {
    outPlaceholder.style.display = 'none';
    outResults.style.display = 'none';
    outLoading.style.display = 'block';
    btnStartScan.disabled = true;

    const formData = new FormData();
    formData.append('image', file);

    try {
      const res = await fetch('/api/predict', { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Classification failed');

      renderOutput(data);
    } catch (err) {
      alert(`Error: ${err.message}`);
      outPlaceholder.style.display = 'block';
    } finally {
      outLoading.style.display = 'none';
      btnStartScan.disabled = false;
    }
  }

  // Sample Thumbnail Item Click
  thumbCards.forEach(card => {
    card.addEventListener('click', async () => {
      const sampleCls = card.getAttribute('data-sample');
      const imgElem = card.querySelector('img');
      const imgSrc = imgElem ? imgElem.src : '';

      switchTab('upload');
      imagePreview.src = imgSrc;
      previewDisplayBox.style.display = 'block';
      fileDropBox.style.display = 'none';

      outPlaceholder.style.display = 'none';
      outResults.style.display = 'none';
      outLoading.style.display = 'block';

      try {
        const res = await fetch(`/api/sample_predict/${sampleCls}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Sample prediction failed');

        if (data.sample_image_url) {
          imagePreview.src = data.sample_image_url;
        }
        renderOutput(data);
      } catch (err) {
        alert(`Sample error: ${err.message}`);
        outPlaceholder.style.display = 'block';
      } finally {
        outLoading.style.display = 'none';
      }
    });
  });

  function renderOutput(data) {
    const cls = data.predicted_class;
    const meta = BIN_MAPPING[cls] || { binId: 'binPurple', emoji: '📦', name: 'Paperboard Bin', cls: 'paper' };

    // Trigger Illustrated Bin Animation & Particles
    openBinAnimation(meta.binId);

    resCardBanner.setAttribute('data-target-cls', meta.cls);
    resBigEmoji.textContent = meta.emoji;
    resClsName.textContent = cls;
    resBinTagPill.textContent = meta.name;
    resScoreDigits.textContent = `${data.confidence}%`;

    if (data.guidance) {
      adviceTitleText.textContent = `${data.guidance.category} — ${data.guidance.action}`;
      adviceText.textContent = data.guidance.tip;
    }

    // Probability Bars
    probBarsFlex.innerHTML = '';
    data.top_predictions.forEach(item => {
      const row = document.createElement('div');
      row.className = 'prob-item-line';
      row.innerHTML = `
        <div class="prob-item-label">
          <span>${BIN_MAPPING[item.class] ? BIN_MAPPING[item.class].emoji : ''} ${item.class}</span>
          <span>${item.prob}%</span>
        </div>
        <div class="prob-item-bg">
          <div class="prob-item-fill" data-cls="${BIN_MAPPING[item.class] ? BIN_MAPPING[item.class].cls : 'paper'}" style="width: 0%"></div>
        </div>
      `;
      probBarsFlex.appendChild(row);

      setTimeout(() => {
        row.querySelector('.prob-item-fill').style.width = `${item.prob}%`;
      }, 50);
    });

    outResults.style.display = 'block';
  }

  // Open Bin Lid & Sparkles Animation
  function openBinAnimation(binId) {
    closeAllBins();
    const targetBin = document.getElementById(binId);
    if (!targetBin) return;

    targetBin.classList.add('active-highlight', 'open-lid');

    // Create floating sparkles
    for (let i = 0; i < 3; i++) {
      const sparkle = document.createElement('span');
      sparkle.className = 'sparkle-particle';
      sparkle.textContent = '✨';
      sparkle.style.left = `${20 + i * 30}%`;
      targetBin.appendChild(sparkle);

      setTimeout(() => sparkle.remove(), 1200);
    }
  }

  function closeAllBins() {
    illustratedBins.forEach(b => {
      b.classList.remove('active-highlight', 'open-lid');
      const particles = b.querySelectorAll('.sparkle-particle');
      particles.forEach(p => p.remove());
    });
  }

  // Copy Tip Action
  btnCopyAdvice.addEventListener('click', () => {
    const text = `${adviceTitleText.textContent}\n${adviceText.textContent}`;
    navigator.clipboard.writeText(text).then(() => {
      const orig = btnCopyAdvice.innerHTML;
      btnCopyAdvice.innerHTML = '✓ Copied!';
      setTimeout(() => btnCopyAdvice.innerHTML = orig, 2000);
    });
  });

  // Eco Impact Calculator Live Updates
  if (inputItemCount) {
    const updateCalculator = () => {
      const count = Math.max(1, parseInt(inputItemCount.value) || 1);
      const co2 = (count * 0.5).toFixed(1);
      const kwh = (count * 1.4).toFixed(1);
      const landfill = (count * 2.5).toFixed(1);

      valCo2.textContent = `${co2} kg`;
      valKwh.textContent = `${kwh} kWh`;
      valLandfill.textContent = `${landfill} L`;
    };

    inputItemCount.addEventListener('input', updateCalculator);
    updateCalculator();
  }

  // FAQ Accordion Toggle
  faqItemCards.forEach(card => {
    const header = card.querySelector('.faq-item-header');
    header.addEventListener('click', () => {
      const isOpen = card.classList.contains('open');
      faqItemCards.forEach(c => c.classList.remove('open'));
      if (!isOpen) card.classList.add('open');
    });
  });

  // -------------------------------------------------------------
  // ANIMATED BACKGROUND PATTERN CANVAS RENDERER
  // -------------------------------------------------------------
  const patternCanvas = document.getElementById('patternCanvas');
  if (patternCanvas) {
    const ctx = patternCanvas.getContext('2d');
    let width = 0, height = 0;
    let particles = [];
    let mouseX = -1000, mouseY = -1000;

    function resizeCanvas() {
      const parent = patternCanvas.parentElement;
      width = patternCanvas.width = parent.offsetWidth || window.innerWidth;
      height = patternCanvas.height = parent.offsetHeight || window.innerHeight;
      createParticles();
    }

    class PatternParticle {
      constructor() {
        this.reset();
      }

      reset() {
        this.x = Math.random() * width;
        this.y = Math.random() * height;
        this.vx = (Math.random() - 0.5) * 0.4;
        this.vy = (Math.random() - 0.5) * 0.4;
        this.radius = Math.random() * 2.5 + 1.5;
        this.type = Math.random() > 0.85 ? 'leaf' : 'dot';
        const colors = [
          'rgba(233, 159, 193, 0.4)',   // Soft Rose Pink (#e99fc1)
          'rgba(150, 240, 227, 0.45)',  // Aqua Mint Cyan (#96f0e3)
          'rgba(107, 68, 58, 0.35)',    // Deep Cocoa (#6b443a)
          'rgba(237, 215, 194, 0.4)'   // Nude Beige (#edd7c2)
        ];
        this.color = colors[Math.floor(Math.random() * colors.length)];
        this.alpha = Math.random() * 0.4 + 0.2;
      }

      update() {
        this.x += this.vx;
        this.y += this.vy;

        // Subtle mouse interaction
        const dx = mouseX - this.x;
        const dy = mouseY - this.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          this.x -= (dx / dist) * 0.6;
          this.y -= (dy / dist) * 0.6;
        }

        // Screen boundary wrap
        if (this.x < 0) this.x = width;
        if (this.x > width) this.x = 0;
        if (this.y < 0) this.y = height;
        if (this.y > height) this.y = 0;
      }

      draw() {
        ctx.save();
        if (this.type === 'leaf') {
          ctx.fillStyle = this.color;
          ctx.beginPath();
          ctx.ellipse(this.x, this.y, this.radius * 2, this.radius, Math.PI / 4, 0, Math.PI * 2);
          ctx.fill();
        } else {
          ctx.fillStyle = this.color;
          ctx.beginPath();
          ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      }
    }

    function createParticles() {
      const particleCount = Math.floor((width * height) / 24000) || 35;
      particles = [];
      for (let i = 0; i < Math.min(particleCount, 60); i++) {
        particles.push(new PatternParticle());
      }
    }

    function drawLines() {
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 110) {
            ctx.strokeStyle = `rgba(150, 240, 227, ${0.25 * (1 - dist / 110)})`;
            ctx.lineWidth = 0.8;
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
          }
        }
      }
    }

    function renderPatternAnimation() {
      ctx.clearRect(0, 0, width, height);
      particles.forEach(p => {
        p.update();
        p.draw();
      });
      drawLines();
      requestAnimationFrame(renderPatternAnimation);
    }

    window.addEventListener('resize', resizeCanvas);
    window.addEventListener('mousemove', (e) => {
      const rect = patternCanvas.getBoundingClientRect();
      mouseX = e.clientX - rect.left;
      mouseY = e.clientY - rect.top;
    });

    resizeCanvas();
    renderPatternAnimation();
  }
});

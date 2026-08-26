document.addEventListener("DOMContentLoaded", () => {
  // Navigation / Tabs System Elements
  const tabAbout = document.getElementById("tabAbout");
  const tabDashboard = document.getElementById("tabDashboard");
  const aboutView = document.getElementById("aboutView");
  const dashboardView = document.getElementById("dashboardView");
  const launchConsoleBtn = document.getElementById("launchConsoleBtn");

  // Form Elements
  const txnForm = document.getElementById("txnForm");
  const analyzeBtn = document.getElementById("analyzeBtn");

  // Threshold Slider Elements
  const sliderLow = document.getElementById("slider-low");
  const sliderMed = document.getElementById("slider-med");
  const lowValBadge = document.getElementById("low-val");
  const medValBadge = document.getElementById("med-val");

  // Assessment Results UI Elements
  const resultEmpty = document.getElementById("resultEmpty");
  const resultBody = document.getElementById("resultBody");
  const gaugeScore = document.getElementById("gaugeScore");
  const gaugeFill = document.getElementById("gaugeFill");
  const riskLevelBadge = document.getElementById("riskLevelBadge");
  const decisionStamp = document.getElementById("decisionStamp");

  const mlValueText = document.getElementById("mlValue");
  const mlBar = document.getElementById("mlBar");
  const ruleValueText = document.getElementById("ruleValue");
  const ruleBar = document.getElementById("ruleBar");

  const reasonsList = document.getElementById("reasonsList");
  const warningsBlock = document.getElementById("warningsBlock");
  const warningsList = document.getElementById("warningsList");

  // System Ledger Metrics Counters
  const mTotal = document.getElementById("mTotal");
  const mLow = document.getElementById("mLow");
  const mMedium = document.getElementById("mMedium");
  const mHigh = document.getElementById("mHigh");
  const mBlocked = document.getElementById("mBlocked");
  const statusDot = document.getElementById("statusDot");
  const statusLabel = document.getElementById("statusLabel");
  const refreshHistoryBtn = document.getElementById("refreshHistory");
  const historyBody = document.getElementById("historyBody");

  // Track operational variables
  let historyData = [];
  const svgCircleRadius = 68;
  const svgCircumference = 2 * Math.PI * svgCircleRadius; // Approx 427.25

  // --- Initial Page Setup ---
  if (gaugeFill) {
    gaugeFill.style.strokeDasharray = svgCircumference;
    gaugeFill.style.strokeDashoffset = svgCircumference;
  }
  updateEngineStatus(true);

  // --- View Switcher Logic ---
  function switchToView(viewName) {
    if (viewName === "dashboard") {
      tabAbout.classList.remove("active");
      tabDashboard.classList.add("active");
      aboutView.classList.remove("active");
      dashboardView.classList.add("active");
    } else {
      tabDashboard.classList.remove("active");
      tabAbout.classList.add("active");
      dashboardView.classList.remove("active");
      aboutView.classList.add("active");
    }
  }

  if (tabAbout) tabAbout.addEventListener("click", () => switchToView("about"));
  if (tabDashboard) tabDashboard.addEventListener("click", () => switchToView("dashboard"));
  if (launchConsoleBtn) {
    launchConsoleBtn.addEventListener("click", () => switchToView("dashboard"));
  }

  // --- Slider Live Value Badge Synchronizers ---
  if (sliderLow && lowValBadge) {
    sliderLow.addEventListener("input", () => {
      lowValBadge.textContent = sliderLow.value;
    });
  }
  if (sliderMed && medValBadge) {
    sliderMed.addEventListener("input", () => {
      medValBadge.textContent = sliderMed.value;
    });
  }

  // --- CORE RISK CALCULATION ENGINE ---
  if (txnForm) {
    txnForm.addEventListener("submit", (event) => {
      event.preventDefault();

      // Retrieve input values safely
      const amount = parseFloat(document.getElementById("amount")?.value) || 0;
      const paymentMethod = document.getElementById("payment_method")?.value || "N/A";
      const deviceType = document.getElementById("device_type")?.value || "N/A";
      const location = document.getElementById("location")?.value || "N/A";
      const velocity = parseInt(document.getElementById("transactions_last_hour")?.value) || 0;
      const accountAge = parseInt(document.getElementById("account_age_days")?.value) || 0;
      const fraudFlags = parseInt(document.getElementById("previous_fraud_count")?.value) || 0;
      const isNewDevice = document.getElementById("new_device")?.checked || false;
      const isLocationVariance = document.getElementById("location_change")?.checked || false;

      // 1. Calculate Rule/Heuristic Base Vectors
      let ruleScore = 0;
      let vectors = [];
      let discrepancies = [];

      if (amount > 50000) {
        ruleScore += 25;
        vectors.push(`High Transaction Value (₹${amount}) triggered upper monitoring tier.`);
      }
      if (velocity > 3) {
        ruleScore += 20;
        vectors.push(`Velocity Spike: ${velocity} transactions processed within 1 hr standard deviation threshold.`);
      }
      if (accountAge < 30) {
        ruleScore += 15;
        vectors.push(`Fresh Account Ledger footprint (<30 days incubation maturity).`);
      }
      if (fraudFlags > 0) {
        ruleScore += 30;
        vectors.push(`System Flag match: Account possesses ${fraudFlags} historical security flags.`);
      }
      if (isNewDevice) {
        ruleScore += 15;
        discrepancies.push("Hardware Fingerprint Variant: Unrecognized hardware cryptographic key pairs.");
      }
      if (isLocationVariance) {
        ruleScore += 15;
        discrepancies.push("Geographic Displacement Context: Active IP routing mismatch away from regional base.");
      }

      ruleScore = Math.min(ruleScore, 100);

      // 2. Mock Machine Learning Predictive Probability Output
      let mlScore = 10;
      if (amount > 20000 || fraudFlags > 0 || isLocationVariance) {
        mlScore = Math.min(Math.floor(ruleScore * 0.85 + Math.random() * 15), 100);
      } else {
        mlScore = Math.floor(10 + Math.random() * 15);
      }

      // 3. Synthesize Weighted Composite Score
      const totalRiskScore = Math.min(Math.floor((mlScore * 0.6) + (ruleScore * 0.4)), 100);

      // Get current variable control thresholds
      const maxLowThresh = sliderLow ? parseInt(sliderLow.value) : 30;
      const maxMedThresh = sliderMed ? parseInt(sliderMed.value) : 70;

      // Determine outcome level and mitigation routing profiles
      let riskLevel = "LOW RISK";
      let statusClass = "low";
      let designatorAction = "APPROVE";

      if (totalRiskScore > maxMedThresh) {
        riskLevel = "HIGH RISK";
        statusClass = "high";
        designatorAction = "BLOCK";
      } else if (totalRiskScore > maxLowThresh) {
        riskLevel = "MEDIUM RISK";
        statusClass = "med";
        designatorAction = "REVIEW";
      }

      // Render operational updates onto UI components
      displayAssessmentResults(totalRiskScore, riskLevel, designatorAction, mlScore, ruleScore, vectors, discrepancies);

      // Store log snapshot
      const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      historyData.unshift({
        timestamp,
        amount: `₹${amount}`,
        method: paymentMethod,
        location,
        score: totalRiskScore,
        level: riskLevel,
        action: designatorAction,
        statusClass
      });

      updateAuditLogTable();
      updateSystemMetricsLedger();
    });
  }

  // --- Dynamic Dashboard Renderer Helpers ---
  function displayAssessmentResults(score, level, action, mlScore, ruleScore, vectors, discrepancies) {
    if (resultEmpty) resultEmpty.setAttribute("hidden", "true");
    if (resultBody) resultBody.removeAttribute("hidden");

    // Dynamic SVG gauge circular trace sweep calculations
    if (gaugeScore) gaugeScore.textContent = score;
    if (gaugeFill) {
      const offsetTraceValue = svgCircumference - (score / 100) * svgCircumference;
      gaugeFill.style.strokeDashoffset = offsetTraceValue;

      if (action === "BLOCK") {
        gaugeFill.style.stroke = "var(--high-risk, #ef4444)";
      } else if (action === "REVIEW") {
        gaugeFill.style.stroke = "var(--med-risk, #f59e0b)";
      } else {
        gaugeFill.style.stroke = "var(--low-risk, #10b981)";
      }
    }

    // Badge stack stamp descriptors
    if (riskLevelBadge) {
      riskLevelBadge.className = `risk-level-badge ${action.toLowerCase()}`;
      riskLevelBadge.textContent = level;
    }
    
    if (decisionStamp) {
      decisionStamp.className = `stamp ${action.toLowerCase()}`;
      decisionStamp.textContent = action;
    }

    // Component tracking progress columns
    if (mlValueText) mlValueText.textContent = `${mlScore}%`;
    if (mlBar) mlBar.style.width = `${mlScore}%`;
    if (ruleValueText) ruleValueText.textContent = `${ruleScore}%`;
    if (ruleBar) ruleBar.style.width = `${ruleScore}%`;

    // Process explanatory reasoning strings
    if (reasonsList) {
      reasonsList.innerHTML = "";
      if (vectors.length === 0) {
        reasonsList.innerHTML = "<li>All tested parameters conform comfortably inside baseline profiles.</li>";
      } else {
        vectors.forEach(v => {
          const li = document.createElement("li");
          li.textContent = v;
          reasonsList.appendChild(li);
        });
      }
    }

    // Discrepancy Alert block configurations
    if (warningsBlock && warningsList) {
      if (discrepancies.length > 0) {
        warningsBlock.removeAttribute("hidden");
        warningsList.innerHTML = "";
        discrepancies.forEach(w => {
          const li = document.createElement("li");
          li.textContent = w;
          warningsList.appendChild(li);
        });
      } else {
        warningsBlock.setAttribute("hidden", "true");
      }
    }
  }

  function updateAuditLogTable() {
    if (!historyBody) return;

    if (historyData.length === 0) {
      historyBody.innerHTML = `<tr class="empty-row"><td colspan="7">No transactions analyzed yet.</td></tr>`;
      return;
    }

    historyBody.innerHTML = historyData.map(row => `
      <tr>
        <td>${row.timestamp}</td>
        <td><b>${row.amount}</b></td>
        <td>${row.method}</td>
        <td>${row.location}</td>
        <td><span class="table-score">${row.score}</span></td>
        <td><span class="badge-status ${row.statusClass}">${row.level}</span></td>
        <td><span class="badge-action ${row.statusClass}">${row.action}</span></td>
      </tr>
    `).join("");
  }

  function updateSystemMetricsLedger() {
    const total = historyData.length;
    const lowCount = historyData.filter(r => r.level === "LOW RISK").length;
    const medCount = historyData.filter(r => r.level === "MEDIUM RISK").length;
    const highCount = historyData.filter(r => r.level === "HIGH RISK").length;
    const blockedCount = historyData.filter(r => r.action === "BLOCK").length;

    if (mTotal) mTotal.textContent = total;
    if (mLow) mLow.textContent = lowCount;
    if (mMedium) mMedium.textContent = medCount;
    if (mHigh) mHigh.textContent = highCount;
    if (mBlocked) mBlocked.textContent = blockedCount;
  }

  function updateEngineStatus(online) {
    if (!statusDot || !statusLabel) return;
    if (online) {
      statusDot.className = "status-dot online";
      statusLabel.textContent = "Risk Engine Core: ACTIVE v2.4.0";
    } else {
      statusDot.className = "status-dot offline";
      statusLabel.textContent = "Engine Offline";
    }
  }

  // --- Manual Reset Action Trigger ---
  if (refreshHistoryBtn) {
    refreshHistoryBtn.addEventListener("click", () => {
      historyData = [];
      updateAuditLogTable();
      updateSystemMetricsLedger();
      if (resultBody) resultBody.setAttribute("hidden", "true");
      if (resultEmpty) resultEmpty.removeAttribute("hidden");
      if (txnForm) txnForm.reset();
      if (lowValBadge && sliderLow) lowValBadge.textContent = sliderLow.value;
      if (medValBadge && sliderMed) medValBadge.textContent = sliderMed.value;
    });
  }
});
/**
 * AppleSupport AI Agent - Interactive Frontend Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  const queryInput = document.getElementById("customer-query-input");
  const charCountEl = document.getElementById("query-char-count");
  const btnRunAgent = document.getElementById("btn-run-agent");
  const btnClearInput = document.getElementById("btn-clear-input");
  const sampleChipsContainer = document.getElementById("sample-chips-container");

  const sliderConfidence = document.getElementById("slider-confidence");
  const valConfidence = document.getElementById("val-confidence");
  const sliderSimilarity = document.getElementById("slider-similarity");
  const valSimilarity = document.getElementById("val-similarity");

  // Output Cards
  const decisionCard = document.getElementById("card-decision-banner");
  const decisionPlaceholder = document.getElementById("decision-placeholder");
  const decisionContent = document.getElementById("decision-content");
  const decisionActionTitle = document.getElementById("decision-action-title");
  const decisionActionTag = document.getElementById("decision-action-tag");
  const decisionPrimaryReason = document.getElementById("decision-primary-reason");
  const decisionAllReasons = document.getElementById("decision-all-reasons");
  const decisionExplanation = document.getElementById("decision-explanation-text");

  const stageDraftCard = document.getElementById("card-stage-draft");
  const draftReplyText = document.getElementById("draft-reply-text");
  const draftModePill = document.getElementById("draft-mode-pill");
  const draftSafetyPill = document.getElementById("draft-safety-pill");
  const draftGroundingNote = document.getElementById("draft-grounding-note");
  const btnCopyDraft = document.getElementById("btn-copy-draft");

  const stageDiagnosticsCard = document.getElementById("card-stage-diagnostics");
  const intentConfBadge = document.getElementById("intent-conf-badge");
  const intentPrimaryCode = document.getElementById("intent-primary-code");
  const intentReviewFlag = document.getElementById("intent-review-flag");
  const probBarsContainer = document.getElementById("prob-bars-container");

  const retrievalSimBadge = document.getElementById("retrieval-sim-badge");
  const retrievalBandTag = document.getElementById("retrieval-band-tag");
  const retrievedCandidatesContainer = document.getElementById("retrieved-candidates-container");

  const stagePreprocessingCard = document.getElementById("card-stage-preprocessing");
  const normCleanText = document.getElementById("norm-clean-text");

  // Golden Tab elements
  const metricTotal = document.getElementById("metric-total-golden");
  const metricDone = document.getElementById("metric-done-golden");
  const metricPending = document.getElementById("metric-pending-golden");
  const metricFreeze = document.getElementById("metric-freeze-golden");
  const goldenTableBody = document.getElementById("golden-table-body");
  const goldenStatusBadge = document.getElementById("golden-status-badge");

  // State
  let samplesList = [];

  // ==========================================
  // Tab Switching
  // ==========================================
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.add("hidden"));

      btn.classList.add("active");
      const targetId = `pane-${btn.dataset.tab}`;
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.remove("hidden");
      }

      if (btn.dataset.tab === "golden") {
        fetchGoldenData();
      }
    });
  });

  // ==========================================
  // Slider Controls
  // ==========================================
  sliderConfidence.addEventListener("input", (e) => {
    valConfidence.textContent = parseFloat(e.target.value).toFixed(2);
  });

  sliderSimilarity.addEventListener("input", (e) => {
    valSimilarity.textContent = parseFloat(e.target.value).toFixed(2);
  });

  // ==========================================
  // Input Helpers
  // ==========================================
  queryInput.addEventListener("input", () => {
    const len = queryInput.value.length;
    charCountEl.textContent = `${len} character${len === 1 ? '' : 's'}`;
  });

  queryInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      executePipeline();
    }
  });

  btnClearInput.addEventListener("click", () => {
    queryInput.value = "";
    charCountEl.textContent = "0 characters";
    document.querySelectorAll(".sample-chip").forEach(c => c.classList.remove("active"));
  });

  // ==========================================
  // Fetch Quick Samples
  // ==========================================
  async function loadSamples() {
    try {
      const res = await fetch("/api/samples");
      if (!res.ok) return;
      samplesList = await res.json();

      sampleChipsContainer.innerHTML = "";
      samplesList.forEach((sample, idx) => {
        const chip = document.createElement("button");
        chip.className = "sample-chip";
        chip.innerHTML = `
          <span class="chip-dot ${sample.expected_action}"></span>
          <span>${sample.title}</span>
        `;
        chip.title = sample.description;
        chip.addEventListener("click", () => {
          document.querySelectorAll(".sample-chip").forEach(c => c.classList.remove("active"));
          chip.classList.add("active");
          queryInput.value = sample.text;
          charCountEl.textContent = `${sample.text.length} characters`;
          executePipeline();
        });
        sampleChipsContainer.appendChild(chip);
      });
    } catch (err) {
      console.error("Failed to load samples:", err);
    }
  }

  // ==========================================
  // Execute Pipeline
  // ==========================================
  btnRunAgent.addEventListener("click", executePipeline);

  async function executePipeline() {
    const text = queryInput.value.trim();
    if (!text) {
      queryInput.focus();
      return;
    }

    const confThreshold = parseFloat(sliderConfidence.value);
    const simThreshold = parseFloat(sliderSimilarity.value);

    // Set loading state
    btnRunAgent.disabled = true;
    btnRunAgent.innerHTML = `
      <span class="pulse-dot" style="width:12px;height:12px;"></span>
      <span>Analyzing Inquiry...</span>
    `;

    try {
      const resp = await fetch("/api/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          confidence_threshold: confThreshold,
          similarity_threshold: simThreshold,
          top_k: 3
        })
      });

      if (!resp.ok) {
        const errData = await resp.json();
        alert(`Error: ${errData.error || "Inference failed"}`);
        return;
      }

      const result = await resp.json();
      renderPipelineResults(result);

    } catch (err) {
      console.error("Pipeline request error:", err);
      alert("Failed to connect to agent server.");
    } finally {
      btnRunAgent.disabled = false;
      btnRunAgent.innerHTML = `
        <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
        <span>Execute End-to-End Pipeline</span>
      `;
    }
  }

  // ==========================================
  // Render Pipeline Results
  // ==========================================
  function renderPipelineResults(data) {
    // Reveal all output sections
    decisionPlaceholder.classList.add("hidden");
    decisionContent.classList.remove("hidden");
    stageDraftCard.classList.remove("hidden");
    stageDiagnosticsCard.classList.remove("hidden");
    stagePreprocessingCard.classList.remove("hidden");

    // 1. Stage 1 Preprocessing
    normCleanText.textContent = data.query.clean_text;

    // 2. Stage 5 Decision Banner
    const isAutoHandle = data.routing.action === "auto_handle";
    decisionCard.className = `glass-card decision-banner-card ${data.routing.action}`;

    decisionActionTitle.textContent = isAutoHandle 
      ? "AUTO-HANDLE CANDIDATE" 
      : "ESCALATE TO HUMAN SPECIALIST";

    decisionActionTag.textContent = isAutoHandle ? "ELIGIBLE" : "ESCALATE";
    decisionActionTag.style.color = isAutoHandle ? "#34d399" : "#fb7185";

    decisionPrimaryReason.textContent = `Reason: ${data.routing.primary_reason_code}`;
    
    if (data.routing.all_reason_codes && data.routing.all_reason_codes !== data.routing.primary_reason_code) {
      decisionAllReasons.textContent = `All Triggers: ${data.routing.all_reason_codes}`;
      decisionAllReasons.classList.remove("hidden");
    } else {
      decisionAllReasons.classList.add("hidden");
    }

    decisionExplanation.textContent = data.routing.decision_explanation;

    // 3. Stage 4 Draft Reply
    draftReplyText.textContent = data.reply_draft.draft_reply;
    draftModePill.textContent = `Mode: ${data.reply_draft.draft_mode}`;

    if (data.reply_draft.restricted_draft) {
      draftSafetyPill.className = "safety-pill restricted";
      draftSafetyPill.textContent = `Restricted: ${data.reply_draft.safety_flags || 'Active Risk'}`;
    } else {
      draftSafetyPill.className = "safety-pill";
      draftSafetyPill.textContent = "Safety: Clean";
    }

    draftGroundingNote.textContent = data.reply_draft.grounding_note;

    // 4. Stage 2 Intent Classifier Diagnostics
    const topIntent = data.intent.predicted_intent;
    const topConfPct = (data.intent.predicted_confidence * 100).toFixed(1);

    intentConfBadge.textContent = `${topConfPct}% Conf`;
    intentPrimaryCode.textContent = topIntent;

    if (data.intent.needs_human_review) {
      intentReviewFlag.classList.remove("hidden");
    } else {
      intentReviewFlag.classList.add("hidden");
    }

    // Probability Bars
    probBarsContainer.innerHTML = "";
    const allProbs = data.intent.all_probabilities || {};
    Object.entries(allProbs).forEach(([intentName, probVal], idx) => {
      const pct = (probVal * 100).toFixed(1);
      const isTop = idx === 0;

      const row = document.createElement("div");
      row.className = `prob-row ${isTop ? 'top' : ''}`;
      row.innerHTML = `
        <div class="prob-label-row">
          <span>${intentName}</span>
          <span class="prob-pct">${pct}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-bar" style="width: ${pct}%"></div>
        </div>
      `;
      probBarsContainer.appendChild(row);
    });

    // 5. Stage 3 Retrieval Diagnostics
    const bestSim = data.retrieval.best_similarity_score;
    retrievalSimBadge.textContent = `Best Cosine: ${bestSim.toFixed(4)}`;
    retrievalBandTag.textContent = data.retrieval.lexical_similarity_band;

    retrievedCandidatesContainer.innerHTML = "";
    const candidates = data.retrieval.candidates || [];

    if (candidates.length === 0) {
      retrievedCandidatesContainer.innerHTML = `<p style="font-size:0.8rem;color:var(--text-muted);">No historical candidates found.</p>`;
    } else {
      candidates.forEach((cand) => {
        const card = document.createElement("div");
        card.className = "candidate-card";
        card.innerHTML = `
          <div class="candidate-header">
            <span>Rank #${cand.rank} • ID: ${cand.retrieved_customer_tweet_id}</span>
            <span>Sim: ${cand.similarity_score.toFixed(4)}</span>
          </div>
          <div class="candidate-query">
            <strong>Customer:</strong> ${escapeHtml(cand.retrieved_customer_text_clean)}
          </div>
          <div class="candidate-reply">
            <strong>Historical @AppleSupport:</strong> ${escapeHtml(cand.retrieved_brand_reply_clean)}
          </div>
        `;
        retrievedCandidatesContainer.appendChild(card);
      });
    }

    // Smooth scroll to decision if mobile
    if (window.innerWidth < 1024) {
      decisionCard.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  // Copy Draft Reply
  btnCopyDraft.addEventListener("click", () => {
    const text = draftReplyText.textContent;
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      const span = btnCopyDraft.querySelector("span");
      const originalText = span.textContent;
      span.textContent = "Copied!";
      setTimeout(() => { span.textContent = originalText; }, 2000);
    });
  });

  // ==========================================
  // Golden Tab Fetch & Render
  // ==========================================
  async function fetchGoldenData() {
    try {
      const [statusRes, rowsRes] = await Promise.all([
        fetch("/api/golden/status"),
        fetch("/api/golden/rows")
      ]);

      if (statusRes.ok) {
        const status = await statusRes.json();
        metricTotal.textContent = status.total_rows;
        metricDone.textContent = status.done_count;
        metricPending.textContent = status.pending_count;
        metricFreeze.textContent = status.is_frozen ? "Frozen (Certified)" : "Guarded";
        goldenStatusBadge.textContent = `${status.completion_percentage}%`;
      }

      if (rowsRes.ok) {
        const data = await rowsRes.json();
        goldenTableBody.innerHTML = "";
        data.rows.forEach(row => {
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td><code>${row.golden_id}</code></td>
            <td style="max-width:320px;">${escapeHtml(row.customer_text_clean)}</td>
            <td>${row.annotator_a_intent || '<em>blank</em>'}</td>
            <td>${row.annotator_b_intent || '<em>blank</em>'}</td>
            <td><strong>${row.final_intent || '<em>pending</em>'}</strong></td>
            <td><span class="status-pill">${row.final_action || 'pending'}</span></td>
            <td><span class="band-tag">${row.final_status}</span></td>
          `;
          goldenTableBody.appendChild(tr);
        });
      }
    } catch (err) {
      console.error("Failed to load golden benchmark:", err);
    }
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/[&<>"']/g, (m) => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#039;'
    }[m]));
  }

  // Initialize samples on load
  loadSamples();
});

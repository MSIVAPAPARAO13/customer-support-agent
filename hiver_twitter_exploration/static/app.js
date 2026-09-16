/**
 * Hiver AI - AppleSupport Agent Dashboard Controller
 * Connects frontend Stitch design components directly to the running backend APIs
 */

document.addEventListener("DOMContentLoaded", () => {
  // Navigation Tabs
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  // Inputs & Controls
  const queryInput = document.getElementById("customer-query-input");
  const charCountEl = document.getElementById("query-char-count");
  const btnRunAgent = document.getElementById("btn-run-agent");
  const btnClearInput = document.getElementById("btn-clear-input");
  const sampleChipsContainer = document.getElementById("sample-chips-container");
  const telemetryLatency = document.getElementById("telemetry-latency");

  // Thresholds
  const sliderConfidence = document.getElementById("slider-confidence");
  const valConfidence = document.getElementById("val-confidence");
  const sliderSimilarity = document.getElementById("slider-similarity");
  const valSimilarity = document.getElementById("val-similarity");

  // Decision Routing Outputs
  const decisionCard = document.getElementById("card-decision-banner");
  const decisionPlaceholder = document.getElementById("decision-placeholder");
  const decisionContent = document.getElementById("decision-content");
  const decisionIcon = document.getElementById("decision-icon");
  const decisionActionTitle = document.getElementById("decision-action-title");
  const decisionActionTag = document.getElementById("decision-action-tag");
  const decisionPrimaryReason = document.getElementById("decision-primary-reason");
  const decisionExplanation = document.getElementById("decision-explanation-text");
  const decisionAllReasons = document.getElementById("decision-all-reasons");

  // Intent Outputs
  const intentConfBadge = document.getElementById("intent-conf-badge");
  const intentPrimaryCode = document.getElementById("intent-primary-code");
  const intentReviewFlag = document.getElementById("intent-review-flag");
  const probBarsContainer = document.getElementById("prob-bars-container");

  // Draft Outputs
  const draftReplyText = document.getElementById("draft-reply-text");
  const draftModePill = document.getElementById("draft-mode-pill");
  const draftSafetyPill = document.getElementById("draft-safety-pill");
  const draftGroundingNote = document.getElementById("draft-grounding-note");
  const btnCopyDraft = document.getElementById("btn-copy-draft");

  // Retrieval Outputs
  const retrievalSimBadge = document.getElementById("retrieval-sim-badge");
  const retrievalBandTag = document.getElementById("retrieval-band-tag");
  const retrievedCandidatesContainer = document.getElementById("retrieved-candidates-container");

  // Golden Tab Elements
  const metricTotal = document.getElementById("metric-total-golden");
  const metricDone = document.getElementById("metric-done-golden");
  const metricPending = document.getElementById("metric-pending-golden");
  const metricFreeze = document.getElementById("metric-freeze-golden");
  const goldenTableBody = document.getElementById("golden-table-body");
  const goldenStatusBadge = document.getElementById("golden-status-badge");

  // HITL Adjudication Bar Elements
  const btnHitlConfirm = document.getElementById("btn-hitl-confirm");
  const btnHitlEscalate = document.getElementById("btn-hitl-escalate");
  const hitlReasonSelect = document.getElementById("hitl-reason-select");
  const btnHitlSave = document.getElementById("btn-hitl-save");

  let currentInferenceResult = null;
  let samplesList = [];

  // ==========================================
  // 1. Navigation Tab Switching
  // ==========================================
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");

      tabBtns.forEach(b => {
        b.classList.remove("bg-indigo-600", "text-white", "shadow-sm");
        b.classList.add("text-gray-400");
      });
      btn.classList.add("bg-indigo-600", "text-white", "shadow-sm");
      btn.classList.remove("text-gray-400");

      tabPanes.forEach(pane => pane.classList.add("hidden"));
      const targetPane = document.getElementById(`pane-${targetTab}`);
      if (targetPane) targetPane.classList.remove("hidden");

      if (targetTab === "golden") {
        fetchGoldenData();
      }
    });
  });

  // ==========================================
  // 2. Threshold Sliders
  // ==========================================
  if (sliderConfidence && valConfidence) {
    sliderConfidence.addEventListener("input", (e) => {
      valConfidence.textContent = parseFloat(e.target.value).toFixed(2);
    });
  }

  if (sliderSimilarity && valSimilarity) {
    sliderSimilarity.addEventListener("input", (e) => {
      valSimilarity.textContent = parseFloat(e.target.value).toFixed(2);
    });
  }

  // ==========================================
  // 3. Input Text Tracking & Clear
  // ==========================================
  if (queryInput && charCountEl) {
    queryInput.addEventListener("input", () => {
      charCountEl.textContent = `${queryInput.value.length} chars`;
    });

    queryInput.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        executePipeline();
      }
    });
  }

  if (btnClearInput) {
    btnClearInput.addEventListener("click", () => {
      queryInput.value = "";
      charCountEl.textContent = "0 chars";
      queryInput.focus();
    });
  }

  // ==========================================
  // 4. Fetch Samples & Setup Preset Pills
  // ==========================================
  async function loadSamples() {
    try {
      const res = await fetch("/api/samples");
      if (res.ok) {
        samplesList = await res.json();
        renderSamplePills(samplesList);
      }
    } catch (err) {
      console.warn("Could not fetch remote samples, using pre-rendered ones", err);
    }
  }

  function renderSamplePills(samples) {
    if (!sampleChipsContainer || !samples.length) return;
    sampleChipsContainer.innerHTML = "";

    samples.forEach(sample => {
      const btn = document.createElement("button");
      btn.type = "button";
      const isCritical = sample.expected_action === "escalate";
      btn.className = isCritical
        ? "preset-pill px-2.5 py-1 text-xs bg-red-500/10 hover:bg-red-500/20 text-red-300 border border-red-500/20 rounded-lg transition-all flex items-center gap-1 font-medium"
        : "preset-pill px-2.5 py-1 text-xs bg-gray-800/80 hover:bg-gray-700/80 text-gray-300 hover:text-white rounded-lg border border-white/5 transition-all flex items-center gap-1";

      btn.innerHTML = `${sample.title}`;
      btn.addEventListener("click", () => {
        queryInput.value = sample.text;
        charCountEl.textContent = `${sample.text.length} chars`;
        executePipeline();
      });
      sampleChipsContainer.appendChild(btn);
    });
  }

  // ==========================================
  // 5. Execute End-to-End Pipeline
  // ==========================================
  if (btnRunAgent) {
    btnRunAgent.addEventListener("click", executePipeline);
  }

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
      <span class="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
      <span>Inference Running...</span>
    `;
    const startTime = performance.now();

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
      currentInferenceResult = result;
      const elapsed = Math.round(performance.now() - startTime);
      if (telemetryLatency) telemetryLatency.textContent = `Latency: ${elapsed}ms`;

      renderPipelineResults(result);

    } catch (err) {
      console.error("Pipeline request error:", err);
      alert("Failed to connect to agent server.");
    } finally {
      btnRunAgent.disabled = false;
      btnRunAgent.innerHTML = `
        <span class="material-symbols-outlined text-[18px]">auto_awesome</span>
        <span>Execute End-to-End Pipeline</span>
      `;
    }
  }

  // ==========================================
  // 6. Render Results into Stitch UI
  // ==========================================
  function renderPipelineResults(data) {
    decisionPlaceholder.classList.add("hidden");
    decisionContent.classList.remove("hidden");

    // Decision Banner Styling
    const isAutoHandle = data.routing.action === "auto_handle";
    
    if (isAutoHandle) {
      decisionCard.className = "glass-panel p-5 rounded-2xl border border-emerald-500/40 shadow-xl bg-emerald-950/20";
      decisionIcon.className = "material-symbols-outlined text-[24px] text-emerald-400";
      decisionIcon.textContent = "check_circle";
      decisionActionTitle.textContent = "AUTO-HANDLE CANDIDATE";
      decisionActionTag.className = "px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30";
      decisionActionTag.textContent = "AUTO-ELIGIBLE";
      decisionPrimaryReason.className = "text-xs font-mono font-semibold text-emerald-400";
    } else {
      const isCritical = data.reply_draft.restricted_draft;
      decisionCard.className = isCritical
        ? "glass-panel p-5 rounded-2xl border border-red-500/40 shadow-xl bg-red-950/20"
        : "glass-panel p-5 rounded-2xl border border-amber-500/40 shadow-xl bg-amber-950/20";
      decisionIcon.className = `material-symbols-outlined text-[24px] ${isCritical ? 'text-red-400' : 'text-amber-400'}`;
      decisionIcon.textContent = isCritical ? "shield" : "report_problem";
      decisionActionTitle.textContent = "ESCALATE TO HUMAN SPECIALIST";
      decisionActionTag.className = isCritical
        ? "px-3 py-1 rounded-full text-xs font-bold bg-red-500/20 text-red-300 border border-red-500/30"
        : "px-3 py-1 rounded-full text-xs font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30";
      decisionActionTag.textContent = "ESCALATE";
      decisionPrimaryReason.className = `text-xs font-mono font-semibold ${isCritical ? 'text-red-400' : 'text-amber-400'}`;
    }

    decisionPrimaryReason.textContent = `Reason: ${data.routing.primary_reason_code}`;
    decisionExplanation.textContent = data.routing.decision_explanation;

    if (data.routing.all_reason_codes && data.routing.all_reason_codes !== data.routing.primary_reason_code) {
      decisionAllReasons.textContent = `All Triggers: ${data.routing.all_reason_codes}`;
      decisionAllReasons.classList.remove("hidden");
    } else {
      decisionAllReasons.classList.add("hidden");
    }

    // Intent & Confidence
    const topConfPct = (data.intent.predicted_confidence * 100).toFixed(1);
    intentConfBadge.textContent = `${topConfPct}% Conf`;
    intentPrimaryCode.textContent = data.intent.predicted_intent;

    if (data.intent.needs_human_review) {
      intentReviewFlag.classList.remove("hidden");
    } else {
      intentReviewFlag.classList.add("hidden");
    }

    // Softmax Distribution Bars
    probBarsContainer.innerHTML = "";
    const allProbs = data.intent.all_probabilities || {};
    Object.entries(allProbs).forEach(([intentName, probVal], idx) => {
      const pct = (probVal * 100).toFixed(1);
      const isTop = idx === 0;

      const row = document.createElement("div");
      row.className = "flex flex-col gap-1";
      row.innerHTML = `
        <div class="flex justify-between items-center text-[11px]">
          <span class="${isTop ? 'text-white font-medium' : 'text-gray-400'}">${intentName}</span>
          <span class="font-mono text-xs ${isTop ? 'text-indigo-300 font-bold' : 'text-gray-500'}">${pct}%</span>
        </div>
        <div class="w-full bg-gray-800 h-1.5 rounded-full overflow-hidden">
          <div class="${isTop ? 'bg-indigo-500' : 'bg-gray-600'} h-full rounded-full transition-all duration-500" style="width: ${pct}%"></div>
        </div>
      `;
      probBarsContainer.appendChild(row);
    });

    // Synthesized Reply Draft
    draftReplyText.textContent = data.reply_draft.draft_reply;
    draftModePill.textContent = `Mode: ${data.reply_draft.draft_mode}`;

    if (data.reply_draft.restricted_draft) {
      draftSafetyPill.className = "px-2 py-0.5 text-[10px] font-semibold bg-red-500/20 text-red-300 border border-red-500/30 rounded";
      draftSafetyPill.textContent = `Restricted: ${data.reply_draft.safety_flags || 'Active Risk'}`;
    } else {
      draftSafetyPill.className = "px-2 py-0.5 text-[10px] font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 rounded";
      draftSafetyPill.textContent = "Safety: Clean";
    }

    draftGroundingNote.textContent = data.reply_draft.grounding_note || "";
    draftGroundingNote.title = data.reply_draft.grounding_note || "";

    // Historical Dialogue Retrieval
    const bestSim = data.retrieval.best_similarity_score;
    retrievalSimBadge.textContent = `Sim: ${bestSim.toFixed(4)}`;
    retrievalBandTag.textContent = data.retrieval.lexical_similarity_band;

    retrievedCandidatesContainer.innerHTML = "";
    const candidates = data.retrieval.candidates || [];

    if (candidates.length === 0) {
      retrievedCandidatesContainer.innerHTML = `<p class="text-xs text-gray-500 italic p-3">No historical dialogue matches found.</p>`;
    } else {
      candidates.forEach((cand) => {
        const candCard = document.createElement("div");
        candCard.className = "p-3 bg-gray-900/80 rounded-xl border border-white/5 flex flex-col gap-1.5 shadow-sm";
        candCard.innerHTML = `
          <div class="flex items-center justify-between text-[11px]">
            <span class="px-1.5 py-0.5 bg-indigo-500/20 text-indigo-300 font-bold rounded text-[10px]">#${cand.rank} Match</span>
            <span class="font-mono text-gray-400 font-medium">Sim: ${cand.similarity_score.toFixed(4)}</span>
          </div>
          <p class="text-[11px] text-gray-300 italic leading-snug">"${escapeHtml(cand.retrieved_customer_text_clean)}"</p>
          <div class="p-2 bg-black/40 rounded-lg text-[10px] text-gray-400 font-mono border border-white/5">
            <span class="text-indigo-400 font-semibold">@AppleSupport:</span> ${escapeHtml(cand.retrieved_brand_reply_clean)}
          </div>
        `;
        retrievedCandidatesContainer.appendChild(candCard);
      });
    }
  }

  // ==========================================
  // 7. Copy Reply Draft
  // ==========================================
  if (btnCopyDraft) {
    btnCopyDraft.addEventListener("click", () => {
      const text = draftReplyText.textContent;
      if (!text) return;
      navigator.clipboard.writeText(text).then(() => {
        const span = btnCopyDraft.querySelector("span:last-child");
        const orig = span.textContent;
        span.textContent = "Copied!";
        setTimeout(() => { span.textContent = orig; }, 1800);
      });
    });
  }

  // ==========================================
  // 8. HITL Bottom Bar Action Handlers
  // ==========================================
  if (btnHitlConfirm) {
    btnHitlConfirm.addEventListener("click", () => {
      if (!currentInferenceResult) {
        alert("Run an inference simulation first.");
        return;
      }
      alert(`[Adjudication Certified] Recorded AI verdict: ${currentInferenceResult.routing.action.toUpperCase()}`);
    });
  }

  if (btnHitlEscalate) {
    btnHitlEscalate.addEventListener("click", () => {
      if (!currentInferenceResult) {
        alert("Run an inference simulation first.");
        return;
      }
      alert(`[Human Override] Forced inquiry into Tier-3 Human Escalation Queue.`);
    });
  }

  if (btnHitlSave) {
    btnHitlSave.addEventListener("click", () => {
      if (!currentInferenceResult) {
        alert("Run an inference simulation first.");
        return;
      }
      const reason = hitlReasonSelect.value;
      alert(`Saved inquiry and rationale (${reason}) to Golden Evaluation Manifest.`);
    });
  }

  // ==========================================
  // 9. Fetch & Render Golden Dataset
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
        metricFreeze.textContent = status.is_frozen ? "Certified" : "Guarded";
        goldenStatusBadge.textContent = `${status.completion_percentage}%`;
      }

      if (rowsRes.ok) {
        const data = await rowsRes.json();
        goldenTableBody.innerHTML = "";
        data.rows.forEach(row => {
          const tr = document.createElement("tr");
          tr.className = "hover:bg-white/5 transition-colors";
          tr.innerHTML = `
            <td class="p-3 font-mono text-gray-400 font-medium">${row.golden_id}</td>
            <td class="p-3 max-w-[280px] truncate text-gray-200" title="${escapeHtml(row.customer_text_clean)}">${escapeHtml(row.customer_text_clean)}</td>
            <td class="p-3 text-gray-400">${row.annotator_a_intent || '—'}</td>
            <td class="p-3 text-gray-400">${row.annotator_b_intent || '—'}</td>
            <td class="p-3 font-semibold text-white">${row.final_intent || 'pending'}</td>
            <td class="p-3">
              <span class="px-2 py-0.5 rounded text-[10px] font-bold ${row.final_action === 'auto_handle' ? 'bg-emerald-500/20 text-emerald-300' : 'bg-amber-500/20 text-amber-300'}">
                ${row.final_action || 'pending'}
              </span>
            </td>
            <td class="p-3">
              <span class="px-2 py-0.5 rounded text-[10px] font-mono bg-gray-800 text-gray-300">${row.final_status}</span>
            </td>
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

  // Initialize
  loadSamples();
});

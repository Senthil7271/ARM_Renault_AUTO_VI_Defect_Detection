/**
 * AutoVI Edge: Industrial Vision Dashboard Client Controller.
 * Handles telemetry polling, dataset image selection, mode toggling, and audit log updates.
 */

let activeCategory = "tank_screw";
let currentMode = "camera";
let datasetImages = [];
let currentFilter = "all";
let isOverlayEnabled = true;

document.addEventListener("DOMContentLoaded", () => {
    initClock();
    loadCategories();
    loadDatasetImages();
    startPolling();
});

// -----------------------------------------------------------------------------
// 1. Station Clock
// -----------------------------------------------------------------------------
function initClock() {
    const clockEl = document.getElementById("station-clock");
    function tick() {
        const now = new Date();
        clockEl.innerText = now.toTimeString().split(" ")[0];
    }
    tick();
    setInterval(tick, 1000);
}

// -----------------------------------------------------------------------------
// 2. Categories & Component Switching
// -----------------------------------------------------------------------------
function loadCategories() {
    fetch("/api/categories")
        .then(r => r.json())
        .then(d => {
            activeCategory = d.active;
            document.getElementById("header-cat-name").innerText = activeCategory.toUpperCase();
            
            const group = document.getElementById("category-button-group");
            group.innerHTML = "";
            d.available.forEach(cat => {
                const btn = document.createElement("button");
                btn.className = `cat-select-btn ${cat.id === activeCategory ? "active" : ""}`;
                btn.onclick = () => selectCategory(cat.id);
                btn.innerHTML = `
                    <span class="cat-btn-name">${cat.name}</span>
                    <span class="cat-btn-meta">&tau; = ${cat.tau.toFixed(2)} &bull; ${cat.config}</span>
                `;
                group.appendChild(btn);
            });
        })
        .catch(err => console.error("Failed to load categories:", err));
}

function selectCategory(catId) {
    fetch("/api/category", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ category: catId })
    })
    .then(r => r.json())
    .then(res => {
        if (res.status === "success") {
            activeCategory = catId;
            document.getElementById("header-cat-name").innerText = activeCategory.toUpperCase();
            loadCategories();
            loadDatasetImages();
        }
    });
}

// -----------------------------------------------------------------------------
// 3. Dataset Part Browser & Selection
// -----------------------------------------------------------------------------
function loadDatasetImages() {
    const gallery = document.getElementById("thumbnail-gallery");
    gallery.innerHTML = '<div class="loading-state">Loading category parts...</div>';

    fetch("/api/dataset_images")
        .then(r => r.json())
        .then(d => {
            datasetImages = d.images || [];
            renderDatasetGallery();
        })
        .catch(err => {
            gallery.innerHTML = '<div class="empty-state">No dataset images found on disk.</div>';
        });
}

function filterDataset(filterType, evt) {
    currentFilter = filterType;
    document.querySelectorAll(".filter-chip").forEach(c => c.classList.remove("active"));
    const target = (evt && evt.target) || (window.event && window.event.target);
    if (target) {
        target.classList.add("active");
    }
    renderDatasetGallery();
}

function renderDatasetGallery() {
    const gallery = document.getElementById("thumbnail-gallery");
    gallery.innerHTML = "";

    const filtered = datasetImages.filter(img => {
        if (currentFilter === "good") return img.label === "NOMINAL";
        if (currentFilter === "defect") return img.label === "DEFECT";
        return true;
    });

    if (filtered.length === 0) {
        gallery.innerHTML = '<div class="empty-state">No parts match this filter.</div>';
        return;
    }

    filtered.forEach(img => {
        const card = document.createElement("div");
        card.className = "thumb-card";
        card.onclick = () => selectStaticImage(img.filename, card);

        const badgeClass = img.label === "NOMINAL" ? "badge-nominal" : "badge-defect";

        card.innerHTML = `
            <img class="thumb-img" src="${img.preview_url}" alt="${img.filename}" loading="lazy">
            <div class="thumb-info">
                <span class="thumb-badge ${badgeClass}">${img.label}</span>
                <span class="thumb-name" title="${img.filename}">${img.filename.split("/").pop()}</span>
            </div>
        `;
        gallery.appendChild(card);
    });
}

function selectStaticImage(filename, cardEl) {
    document.querySelectorAll(".thumb-card").forEach(c => c.classList.remove("selected"));
    if (cardEl) cardEl.classList.add("selected");

    fetch("/api/select_image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: filename })
    })
    .then(r => r.json())
    .then(d => {
        if (d.status === "success") {
            setModeDisplay("static");
            updateDashboardTelemetry(d.result);
            const streamImg = document.getElementById("inspection-stream");
            if (streamImg) {
                streamImg.src = "/video_feed?t=" + new Date().getTime();
            }
        }
    })
    .catch(err => console.error("Error inspecting image:", err));
}

// -----------------------------------------------------------------------------
// 4. Mode Switching (Camera / Dataset / Conveyor)
// -----------------------------------------------------------------------------
function switchMode(mode) {
    fetch("/api/mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: mode === "dataset" ? "static" : mode })
    })
    .then(r => r.json())
    .then(d => {
        if (d.status === "success") {
            setModeDisplay(mode);
        }
    });
}

function setModeDisplay(mode) {
    currentMode = mode;
    document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));

    const btnMap = {
        camera: "btn-mode-camera",
        dataset: "btn-mode-dataset",
        static: "btn-mode-dataset",
        conveyor: "btn-mode-conveyor",
    };
    const activeBtn = document.getElementById(btnMap[mode]);
    if (activeBtn) activeBtn.classList.add("active");

    const labelMap = {
        camera: "LIVE OPTICAL SENSOR",
        dataset: "DATASET SELECTOR",
        static: "STATIC PART INSPECTION",
        conveyor: "CONVEYOR SIMULATION",
    };
    document.getElementById("active-mode-label").innerText = labelMap[mode] || "ACTIVE";

    // Show/hide dataset browser panel
    const browserPanel = document.getElementById("dataset-browser-panel");
    if (mode === "dataset" || mode === "static") {
        browserPanel.style.display = "flex";
    } else {
        browserPanel.style.display = "none";
    }
}

// -----------------------------------------------------------------------------
// 5. Overlay Toggle & Shift Reset
// -----------------------------------------------------------------------------
function toggleOverlay() {
    fetch("/api/toggle_overlay", { method: "POST" })
        .then(r => r.json())
        .then(d => {
            isOverlayEnabled = d.overlay_enabled;
            const btn = document.getElementById("btn-toggle-overlay");
            if (isOverlayEnabled) {
                btn.classList.add("active-overlay");
            } else {
                btn.classList.remove("active-overlay");
            }
        });
}

function resetShiftStats() {
    if (confirm("Reset current shift inspection counters to zero?")) {
        fetch("/api/reset_stats", { method: "POST" })
            .then(r => r.json())
            .then(() => {
                pollAuditLog();
            });
    }
}

// -----------------------------------------------------------------------------
// 6. Telemetry & Annunciator Updates
// -----------------------------------------------------------------------------
function startPolling() {
    setInterval(pollTelemetry, 400);
    setInterval(pollAuditLog, 1200);
}

function pollTelemetry() {
    fetch("/api/telemetry")
        .then(r => r.json())
        .then(d => {
            updateDashboardTelemetry(d);
        })
        .catch(err => {});
}

function updateDashboardTelemetry(d) {
    if (!d) return;

    if (!window.initialModeSynced && d.mode) {
        window.initialModeSynced = true;
        setModeDisplay(d.mode);
    }

    // 1. Annunciator Status Banner (PASS / REJECT)
    const banner = document.getElementById("annunciator-banner");
    const title = document.getElementById("annunciator-title");
    const subtext = document.getElementById("annunciator-subtext");
    const scoreTau = document.getElementById("annunciator-score-tau");
    const delta = document.getElementById("annunciator-delta");

    const isPass = (d.verdict === "PASS");

    if (isPass) {
        banner.className = "annunciator-banner banner-pass";
        title.innerText = "PASS [OK]";
        subtext.innerText = `NOMINAL PART SPECIFICATION \u2022 ${d.current_file}`;
    } else {
        banner.className = "annunciator-banner banner-reject";
        title.innerText = "REJECT";
        subtext.innerText = `ANOMALY DETECTED (EXCEEDS TOLERANCE) \u2022 ${d.current_file}`;
    }

    scoreTau.innerText = `${(d.score || 0).toFixed(2)} / ${(d.threshold || 0).toFixed(2)}`;
    
    const margin = (d.margin || 0);
    delta.innerText = (margin > 0 ? `+${margin.toFixed(2)}` : margin.toFixed(2));
    delta.style.color = isPass ? "var(--status-pass-border)" : "var(--status-reject-border)";

    // 2. Anomaly Tolerance Gauge
    const score = d.score || 0;
    const maxGauge = 35.0;
    const fillPct = Math.min(100, Math.max(0, (score / maxGauge) * 100));
    const pinPct = Math.min(100, Math.max(0, ((d.threshold || 17.64) / maxGauge) * 100));

    document.getElementById("gauge-fill-bar").style.width = `${fillPct}%`;
    document.getElementById("gauge-pin").style.left = `${pinPct}%`;
    document.getElementById("gauge-threshold-label").innerText = `Threshold: ${(d.threshold || 0).toFixed(2)}`;
    document.getElementById("telemetry-score").innerText = score.toFixed(4);
    document.getElementById("telemetry-margin").innerText = margin.toFixed(4);

    // 3. Hardware Compute Telemetry
    document.getElementById("backend-status-pill").innerText = d.backend || "ARM Cortex-A76";
    document.getElementById("telemetry-latency").innerText = `${(d.latency_ms || 0).toFixed(1)} ms`;
    document.getElementById("telemetry-npu").innerText = `${(d.npu_ms || 0).toFixed(1)} ms`;
    document.getElementById("telemetry-nn").innerText = `${(d.nn_ms || 0).toFixed(1)} ms`;
    document.getElementById("telemetry-fps").innerText = `${(d.fps || 0).toFixed(1)} FPS`;

    // 4. Shift Quality Yield (KPIs)
    if (d.stats) {
        document.getElementById("kpi-total").innerText = d.stats.total_inspected || 0;
        document.getElementById("kpi-passed").innerText = d.stats.passed || 0;
        document.getElementById("kpi-rejected").innerText = d.stats.rejected || 0;
        document.getElementById("kpi-fpy").innerText = `${d.stats.fpy_pct || 100.0}%`;
    }
}

// -----------------------------------------------------------------------------
// 7. Shift Audit Log Polling
// -----------------------------------------------------------------------------
function pollAuditLog() {
    fetch("/api/audit_log")
        .then(r => r.json())
        .then(d => {
            const tbody = document.getElementById("audit-log-tbody");
            const logs = d.audit_log || [];

            if (logs.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" class="empty-state">No inspection records yet.</td></tr>';
                return;
            }

            tbody.innerHTML = logs.map(entry => {
                const isPass = (entry.verdict === "PASS");
                const badgeClass = isPass ? "badge-pass-sm" : "badge-reject-sm";
                const shortFile = entry.filename.length > 14 ? entry.filename.slice(-14) : entry.filename;

                return `
                    <tr>
                        <td>#${entry.id}</td>
                        <td>${entry.time}</td>
                        <td title="${entry.filename}">${shortFile}</td>
                        <td>${entry.score.toFixed(2)}</td>
                        <td><span class="${badgeClass}">${entry.verdict}</span></td>
                    </tr>
                `;
            }).join("");
        })
        .catch(err => {});
}

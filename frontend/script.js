const formatter = {
    snippet(text, limit = 220) {
        if (!text) return "预览内容为空";
        const clean = text.replace(/\s+/g, " ").trim();
        if (clean.length <= limit) return clean;
        return `${clean.slice(0, limit - 1)}…`;
    },
};

document.addEventListener("DOMContentLoaded", () => {
    const chatLog = document.getElementById("chat-log");
    const form = document.getElementById("query-form");
    // const textarea = document.getElementById("query"); // Replaced by CodeMirror
    const modelSelect = document.getElementById("model");
    const searchToggle = document.getElementById("search-toggle");
    const searchSourceDropdown = document.getElementById("search-source-dropdown");
    const searchSourceButton = document.getElementById("search-source-button");
    const searchSourceMenu = document.getElementById("search-source-menu");
    const statusMessage = document.getElementById("status-message");
    const sendButton = form.querySelector(".send-btn");
    const forceSearchWrapper = document.getElementById("force-search-wrapper");
    const forceSearchButton = document.getElementById("force-search-button");
    const uploadButton = document.getElementById("upload-button");
    const fileInput = document.getElementById("file-input");
    const fileList = document.getElementById("file-list");
    const imagePreviewList = document.getElementById("image-preview-list");
    const totalLimitInput = document.getElementById("search-total-limit");
    const perSourceLimitInput = document.getElementById("search-per-source-limit");
    const referenceLimitInput = document.getElementById("search-reference-limit");
    const searchSourceCheckboxes = searchSourceMenu
        ? Array.from(searchSourceMenu.querySelectorAll('input[type="checkbox"]'))
        : [];

    const timingToggle = document.getElementById("timing-toggle");
    const timingDropdown = document.getElementById("timing-dropdown");
    const timingButton = document.getElementById("timing-button");
    const timingMenu = document.getElementById("timing-menu");
    const timingCheckboxes = timingMenu
        ? Array.from(timingMenu.querySelectorAll('input[type="checkbox"]'))
        : [];

    // Initialize CodeMirror
    const editor = CodeMirror(document.getElementById("query-editor"), {
        mode: "markdown",
        lineNumbers: false,
        lineWrapping: true,
        viewportMargin: Infinity,
        placeholder: "发送您的问题，例如：介绍一下检索增强生成的优势",
        extraKeys: {
            "Ctrl-Enter": function(cm) {
                form.dispatchEvent(new Event("submit", { cancelable: true }));
            },
            "Cmd-Enter": function(cm) {
                form.dispatchEvent(new Event("submit", { cancelable: true }));
            }
        }
    });

    // Sync CodeMirror changes to hidden textarea (optional, but good for form semantics)
    editor.on("change", (cm) => {
        const val = cm.getValue();
        const textarea = document.getElementById("query");
        if (textarea) textarea.value = val;
    });

    const state = {
        loading: false,
        searchSources: new Set(),
        timingOptions: new Set(['total', 'search', 'llm', 'tools']), // 默认全部显示
        forceSearch: false,
        images: [],
    };

    let collapsibleSectionId = 0;
    let loadingInterval = null;

    function startLoadingAnimation(messageEl, isSearchEnabled) {
        const bubble = messageEl.querySelector(".bubble");
        // Remove the CSS-based loader class if it exists, as we are using custom HTML
        messageEl.classList.remove("pending"); 
        
        // Initial Skeleton HTML
        bubble.innerHTML = `
            <div class="skeleton-loader">
                <div class="skeleton-status">正在分析您的提问...</div>
                <div class="skeleton-lines">
                    <div class="skeleton-line" style="width: 92%"></div>
                    <div class="skeleton-line" style="width: 75%"></div>
                    <div class="skeleton-line" style="width: 88%"></div>
                </div>
            </div>
        `;
        
        const statusEl = bubble.querySelector(".skeleton-status");
        const steps = isSearchEnabled 
            ? [
                { time: 1500, text: "正在联网搜索相关信息..." },
                { time: 4500, text: "正在阅读并理解搜索结果..." },
                { time: 8000, text: "正在整合信息并生成回答..." },
                { time: 15000, text: "内容较多，请耐心等待..." }
              ]
            : [
                { time: 1000, text: "正在检索本地知识库..." },
                { time: 3000, text: "正在生成回答..." },
                { time: 8000, text: "正在组织语言..." }
              ];
              
        const startTime = Date.now();
        
        if (loadingInterval) clearInterval(loadingInterval);
        
        loadingInterval = setInterval(() => {
            const elapsed = Date.now() - startTime;
            let currentText = null;
            // Find the latest step that matches the elapsed time
            for (const step of steps) {
                if (elapsed >= step.time) {
                    currentText = step.text;
                }
            }
            
            if (currentText && statusEl && statusEl.textContent !== currentText) {
                statusEl.textContent = currentText;
            }
        }, 500);
    }

    function stopLoadingAnimation() {
        if (loadingInterval) {
            clearInterval(loadingInterval);
            loadingInterval = null;
        }
    }

    function refreshSearchSourceButtonLabel() {
        if (!searchSourceButton) return;
        const count = state.searchSources.size;
        searchSourceButton.textContent = count === searchSourceCheckboxes.length
            ? "搜索源"
            : `搜索源 (${count})`;
    }

    function closeSearchSourceMenu() {
        if (!searchSourceDropdown) return;
        searchSourceDropdown.classList.remove("open");
        if (searchSourceButton) {
            searchSourceButton.setAttribute("aria-expanded", "false");
        }
    }

    function openSearchSourceMenu() {
        if (!searchSourceDropdown) return;
        searchSourceDropdown.classList.add("open");
        if (searchSourceButton) {
            searchSourceButton.setAttribute("aria-expanded", "true");
        }
    }

    function toggleSearchSourceMenu() {
        if (!searchSourceDropdown) return;
        if (searchSourceDropdown.classList.contains("open")) {
            closeSearchSourceMenu();
        } else {
            openSearchSourceMenu();
        }
    }

    function updateSearchSourceVisibility() {
        const shouldShow = searchToggle.checked;
        if (searchSourceDropdown) {
            searchSourceDropdown.classList.toggle("hidden", !shouldShow);
            if (!shouldShow) {
                closeSearchSourceMenu();
            }
        }
        setSearchLimitInputsEnabled(shouldShow);
        updateForceSearchVisibility();
    }

    function setForceSearchState(enabled) {
        state.forceSearch = Boolean(enabled);
        if (forceSearchButton) {
            forceSearchButton.setAttribute("aria-pressed", state.forceSearch ? "true" : "false");
            forceSearchButton.classList.toggle("active", state.forceSearch);
        }
    }

    function updateForceSearchVisibility() {
        if (!forceSearchWrapper) return;
        const shouldShow = searchToggle.checked;
        forceSearchWrapper.classList.toggle("hidden", !shouldShow);
        if (!shouldShow && state.forceSearch) {
            setForceSearchState(false);
        }
    }

    function initializeSearchSources() {
        if (!searchSourceCheckboxes.length) return;
        state.searchSources.clear();
        for (const checkbox of searchSourceCheckboxes) {
            if (checkbox.checked) {
                state.searchSources.add(checkbox.value);
            }
        }
        refreshSearchSourceButtonLabel();
    }

    function handleSearchSourceChange(event) {
        const checkbox = event.target;
        if (!checkbox || !checkbox.value) return;
        const value = checkbox.value;

        if (checkbox.checked) {
            state.searchSources.add(value);
        } else {
            const hasValue = state.searchSources.has(value);
            if (hasValue && state.searchSources.size === 1) {
                checkbox.checked = true;
                statusMessage.textContent = "至少选择一个搜索源";
                return;
            }
            state.searchSources.delete(value);
        }
        refreshSearchSourceButtonLabel();
    }

    function setSearchLimitInputsEnabled(enabled) {
        if (totalLimitInput) {
            totalLimitInput.disabled = !enabled;
        }
        if (perSourceLimitInput) {
            perSourceLimitInput.disabled = !enabled;
        }
        if (referenceLimitInput) {
            referenceLimitInput.disabled = !enabled;
        }
    }

    function refreshTimingButtonLabel() {
        if (!timingButton) return;
        const count = state.timingOptions.size;
        timingButton.textContent = count === timingCheckboxes.length
            ? "时间详情"
            : `时间详情 (${count})`;
    }

    function closeTimingMenu() {
        if (!timingDropdown) return;
        timingDropdown.classList.remove("open");
        if (timingButton) {
            timingButton.setAttribute("aria-expanded", "false");
        }
    }

    function openTimingMenu() {
        if (!timingDropdown) return;
        timingDropdown.classList.add("open");
        if (timingButton) {
            timingButton.setAttribute("aria-expanded", "true");
        }
    }

    function toggleTimingMenu() {
        if (!timingDropdown) return;
        if (timingDropdown.classList.contains("open")) {
            closeTimingMenu();
        } else {
            openTimingMenu();
        }
    }

    function updateTimingVisibility() {
        const shouldShow = timingToggle.checked;
        if (timingDropdown) {
            timingDropdown.classList.toggle("hidden", !shouldShow);
            if (!shouldShow) {
                closeTimingMenu();
            }
        }
    }

    function initializeTimingOptions() {
        if (!timingCheckboxes.length) return;
        state.timingOptions.clear();
        for (const checkbox of timingCheckboxes) {
            if (checkbox.checked) {
                state.timingOptions.add(checkbox.value);
            }
        }
        refreshTimingButtonLabel();
    }

    function handleTimingChange(event) {
        const checkbox = event.target;
        if (!checkbox || !checkbox.value) return;
        const value = checkbox.value;

        if (checkbox.checked) {
            state.timingOptions.add(value);
        } else {
            const hasValue = state.timingOptions.has(value);
            if (hasValue && state.timingOptions.size === 1) {
                checkbox.checked = true;
                statusMessage.textContent = "至少选择一个时间选项";
                return;
            }
            state.timingOptions.delete(value);
        }
        refreshTimingButtonLabel();
    }

    function normalizeSearchLimits() {
        let total = 1;
        if (totalLimitInput) {
            total = parseInt(totalLimitInput.value, 10);
            if (!Number.isFinite(total) || total < 1) {
                total = 1;
            }
            if (total > 30) {
                total = 30;
            }
            totalLimitInput.value = String(total);
        }

        if (perSourceLimitInput) {
            let perSource = parseInt(perSourceLimitInput.value, 10);
            if (!Number.isFinite(perSource) || perSource < 1) {
                perSource = 1;
            }
            if (perSource > 20) {
                perSource = 20;
            }
            if (perSource > total) {
                perSource = total;
            }
            perSourceLimitInput.value = String(perSource);
        }

        if (referenceLimitInput) {
            let referenceLimit = parseInt(referenceLimitInput.value, 10);
            if (!Number.isFinite(referenceLimit) || referenceLimit < 1) {
                referenceLimit = 1;
            }
            if (referenceLimit > 20) {
                referenceLimit = 20;
            }
            referenceLimitInput.value = String(referenceLimit);
        }
    }

    async function loadAvailableModels() {
        // Provider display names and sort order
        const providerMeta = {
            zai: { label: "Zai", order: 1 },
            glm: { label: "GLM", order: 2 },
            openai: { label: "OpenAI", order: 3 },
            anthropic: { label: "Anthropic", order: 4 },
            google: { label: "Google", order: 5 },
            minimax: { label: "Minimax", order: 6 },
            hkgai: { label: "HKGAI", order: 7 },
            openrouter: { label: "OpenRouter", order: 8 },
        };

        function normalizeProvider(name) {
            const key = (name || "").toString().trim().toLowerCase();
            return providerMeta[key] ? key : (key || "openrouter");
        }

        function buildLabel(providerKey, id) {
            const meta = providerMeta[providerKey] || { label: providerKey };
            return `${meta.label} — ${id}`;
        }

        try {
            const response = await fetch("/api/models");
            if (!response.ok) throw new Error("Failed to fetch models");

            const data = await response.json();
            const rawModels = Array.isArray(data.models) ? data.models : [];

            // Deduplicate by id, prefer non-OpenRouter provider when duplicates occur
            const byId = new Map();
            for (const m of rawModels) {
                const id = (m && m.id) ? String(m.id) : null;
                if (!id) continue;
                const providerKey = normalizeProvider(m.provider);
                const existing = byId.get(id);
                if (!existing) {
                    byId.set(id, { id, provider: providerKey });
                } else {
                    // Prefer non-openrouter over openrouter for the same id
                    if (existing.provider === "openrouter" && providerKey !== "openrouter") {
                        byId.set(id, { id, provider: providerKey });
                    }
                }
            }

            // Group by provider and sort
            const groups = new Map();
            for (const { id, provider } of byId.values()) {
                if (!groups.has(provider)) groups.set(provider, []);
                groups.get(provider).push({ id, label: buildLabel(provider, id) });
            }

            // Clear existing options and build optgroups
            modelSelect.innerHTML = "";
            const defaultOption = document.createElement("option");
            defaultOption.value = "";
            defaultOption.textContent = "默认模型 (Zai - glm-4.6)";
            modelSelect.appendChild(defaultOption);

            // Sort providers by defined order, then alphabetical fallback
            const sortedProviders = Array.from(groups.keys()).sort((a, b) => {
                const oa = providerMeta[a]?.order ?? 99;
                const ob = providerMeta[b]?.order ?? 99;
                if (oa !== ob) return oa - ob;
                return (providerMeta[a]?.label || a).localeCompare(providerMeta[b]?.label || b, "zh-Hans-CN");
            });

            for (const p of sortedProviders) {
                const optgroup = document.createElement("optgroup");
                optgroup.label = providerMeta[p]?.label || p;
                const items = groups.get(p).sort((x, y) => x.label.localeCompare(y.label, "zh-Hans-CN"));
                for (const item of items) {
                    const option = document.createElement("option");
                    option.value = item.id;
                    option.textContent = item.label;
                    optgroup.appendChild(option);
                }
                modelSelect.appendChild(optgroup);
            }

            console.log(`Loaded ${byId.size} unique models across ${sortedProviders.length} providers`);
        } catch (error) {
            console.error("Failed to load models:", error);
            // Fallback: minimal, already grouped suggestions
            modelSelect.innerHTML = "";
            const defaultOption = document.createElement("option");
            defaultOption.value = "";
            defaultOption.textContent = "默认模型 (Zai - glm-4.6)";
            modelSelect.appendChild(defaultOption);

            const fallback = {
                Zai: [
                    { id: "glm-4.6", label: "Zai — glm-4.6" },
                    { id: "glm-4.5-air", label: "Zai — glm-4.5-air" },
                ],
                GLM: [
                    { id: "glm", label: "GLM — glm-4.6 (provider default)" },
                ],
                OpenRouter: [
                    { id: "minimax/minimax-m2:free", label: "OpenRouter — minimax/minimax-m2:free" },
                    { id: "deepseek/deepseek-r1-0528:free", label: "OpenRouter — deepseek/deepseek-r1-0528:free" },
                ],
                OpenAI: [
                    { id: "openai", label: "OpenAI — gpt-3.5-turbo (provider default)" },
                ],
                Anthropic: [
                    { id: "anthropic", label: "Anthropic — Claude (provider default)" },
                ],
                Google: [
                    { id: "google", label: "Google — gemini-pro (provider default)" },
                ],
                Minimax: [
                    { id: "minimax", label: "Minimax — minimax-m2:free (provider default)" },
                ],
                HKGAI: [
                    { id: "hkgai", label: "HKGAI — HKGAI-V1 (provider default)" },
                ],
            };

            for (const groupName of ["Zai", "GLM", "OpenAI", "Anthropic", "Google", "Minimax", "HKGAI", "OpenRouter"]) {
                const optgroup = document.createElement("optgroup");
                optgroup.label = groupName;
                for (const item of fallback[groupName]) {
                    const option = document.createElement("option");
                    option.value = item.id;
                    option.textContent = item.label;
                    optgroup.appendChild(option);
                }
                modelSelect.appendChild(optgroup);
            }
        }
    }

    function ensurePlaceholder() {
        if (chatLog.children.length > 0) return;
        chatLog.classList.add("empty");
        const placeholder = document.createElement("div");
        placeholder.className = "placeholder";
        placeholder.textContent = "开始一次对话，支持联网搜索与本地文档检索增强。";
        chatLog.appendChild(placeholder);
    }

    function removePlaceholder() {
        if (!chatLog.classList.contains("empty")) return;
        chatLog.classList.remove("empty");
        const placeholder = chatLog.querySelector(".placeholder");
        if (placeholder) {
            placeholder.remove();
        }
    }

    function scrollToBottom() {
        requestAnimationFrame(() => {
            chatLog.scrollTo({
                top: chatLog.scrollHeight,
                behavior: "smooth",
            });
        });
    }

    function appendMessage(role, content) {
        removePlaceholder();
        const message = document.createElement("article");
        message.className = `message ${role}`;

        const bubble = document.createElement("div");
        bubble.className = "bubble";
        bubble.textContent = content;

        message.appendChild(bubble);
        chatLog.appendChild(message);
        scrollToBottom();

        return message;
    }

    // --- Minimal Markdown rendering (safe subset) ---
    function escapeHTML(str) {
        return (str || "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
    }

    function renderMarkdown(md) {
        if (!md) return "";
        // 1. Escape HTML first to prevent XSS and ensure correct rendering order
        let src = escapeHTML(String(md));

        // 2. Handle fenced code blocks: ```lang\ncode```
        // Regex captures optional language identifier and the code content
        src = src.replace(/```(\w*)\n?([\s\S]*?)```/g, (m, lang, code) => {
            const language = lang ? lang.toLowerCase() : 'none';
            // code is already escaped
            return `<pre><code class="language-${language}">${code}</code></pre>`;
        });

        // 3. Inline code
        src = src.replace(/`([^`]+)`/g, (m, code) => `<code>${code}</code>`);

        // 4. Bold and italic
        src = src.replace(/\*\*([^*]+)\*\*/g, (m, t) => `<strong>${t}</strong>`);
        src = src.replace(/(^|\s)\*([^*]+)\*(?=\s|$)/g, (m, pre, t) => `${pre}<em>${t}</em>`);

        // 5. Links [text](url)
        src = src.replace(/\[([^\]]+)\]\(((?:https?:\/\/)[^\s)]+)\)/g, (m, text, url) =>
            `<a href="${url}" target="_blank" rel="noopener noreferrer">${text}</a>`
        );

        // 6. Headings (#, ##, ###)
        src = src.replace(/^###\s+(.+)$/gm, (m, t) => `<h3>${t}</h3>`);
        src = src.replace(/^##\s+(.+)$/gm, (m, t) => `<h3>${t}</h3>`);
        src = src.replace(/^#\s+(.+)$/gm, (m, t) => `<h3>${t}</h3>`);

        // 7. Unordered list
        src = src.replace(/^(?:\s*[-*]\s.+(?:\n|$))+?/gm, (block) => {
            // Remove the leading marker and wrap in li
            // Note: block is already escaped, so markers are - or *
            const items = block.trim().split(/\n/).map(l => l.replace(/^\s*[-*]\s+/, "").trim());
            const lis = items.map(it => `<li>${it}</li>`).join("");
            return `<ul>${lis}</ul>`;
        });

        // 8. Paragraphs
        // Split by double newlines, but ignore if it's already an HTML block we created
        const parts = src.split(/\n{2,}/).map(p => {
            if (/^\s*<(?:h3|ul|pre)/.test(p)) return p;
            const withBr = p.replace(/\n/g, "<br>");
            return `<p>${withBr}</p>`;
        });

        return parts.join("");
    }

    function createBadge(text) {
        const badge = document.createElement("span");
        badge.className = "control-flag";
        badge.textContent = text;
        return badge;
    }

    function createCollapsibleSection(title, contentNode, options = {}) {
        const section = document.createElement("div");
        section.className = "message-extras collapsible-section";
        if (options.sectionClass) {
            section.classList.add(options.sectionClass);
        }

        const header = document.createElement("div");
        header.className = "collapsible-header";

        const heading = document.createElement("h4");
        heading.textContent = title;
        header.appendChild(heading);

        const button = document.createElement("button");
        button.type = "button";
        button.className = "collapse-toggle";
        button.textContent = options.expandLabel || "展开全部";
        button.setAttribute("aria-expanded", "false");
        header.appendChild(button);

        const body = document.createElement("div");
        body.className = "collapsible-body";
        if (options.bodyClass) {
            body.classList.add(options.bodyClass);
        }
        const bodyId = options.bodyId || `collapsible-${++collapsibleSectionId}`;
        body.id = bodyId;
        body.setAttribute("role", options.bodyRole || "region");
        body.setAttribute("aria-hidden", "true");
        button.setAttribute("aria-controls", bodyId);
        body.appendChild(contentNode);

        section.appendChild(header);
        section.appendChild(body);
        section.classList.add("collapsed");

        button.addEventListener("click", () => {
            const isCollapsed = section.classList.toggle("collapsed");
            const expanded = !isCollapsed;
            button.setAttribute("aria-expanded", expanded ? "true" : "false");
            button.textContent = expanded
                ? options.collapseLabel || "收起"
                : options.expandLabel || "展开全部";
            body.setAttribute("aria-hidden", expanded ? "false" : "true");
            section.dispatchEvent(new CustomEvent("collapsible-toggle", {
                bubbles: true,
                detail: { expanded, section },
            }));
        });

        return section;
    }

    function buildTimingExtras(timings, control, searchQuery, retrievedDocs) {
        const safeTimings = (timings && typeof timings === "object") ? timings : {};
        const searchSources = Array.isArray(safeTimings.search_sources) ? safeTimings.search_sources : [];
        const llmCalls = Array.isArray(safeTimings.llm_calls) ? safeTimings.llm_calls : [];
        const toolCalls = Array.isArray(safeTimings.tool_calls) ? safeTimings.tool_calls : [];
        const hasTotal = typeof safeTimings.total_ms === "number";
        
        // Check for keywords
        const keywords = (control && Array.isArray(control.keywords)) ? control.keywords : [];
        const hasKeywords = keywords.length > 0;
        const hasSearchQuery = !!searchQuery;

        // Build badges (control flags)
        const controlFlags = document.createElement("div");
        controlFlags.className = "control-flags";

        if (control.search_allowed === false) {
            controlFlags.appendChild(createBadge("联网：关闭"));
        } else if (control.search_performed) {
            controlFlags.appendChild(createBadge("联网搜索：已触发"));
        } else if (control.search_allowed) {
            controlFlags.appendChild(createBadge("联网搜索：未触发"));
        }

        if (control.force_search_enabled) {
            controlFlags.appendChild(createBadge("强制搜索"));
        }

        if (control.local_docs_present) {
            controlFlags.appendChild(createBadge("本地文档：可用"));
        } else if (Array.isArray(retrievedDocs) && retrievedDocs.length === 0) {
            controlFlags.appendChild(createBadge("本地文档：为空"));
        }

        if (control.hybrid_mode) {
            controlFlags.appendChild(createBadge("混合检索"));
        }

        if (typeof control.search_total_limit === "number") {
            controlFlags.appendChild(createBadge(`汇总上限：${control.search_total_limit}`));
        }

        if (typeof control.search_per_source_limit === "number") {
            controlFlags.appendChild(createBadge(`单源上限：${control.search_per_source_limit}`));
        }

        if (typeof control.search_reference_limit === "number") {
            controlFlags.appendChild(createBadge(`参考链接：${control.search_reference_limit}`));
        }
        
        // Add Domain Intelligence Badge
        if (safeTimings && typeof safeTimings['领域智能类型'] === 'string') {
            controlFlags.appendChild(createBadge(`领域智能类型：${safeTimings['领域智能类型']}`));
        }

        const hasBadges = controlFlags.children.length > 0;

        if (!hasTotal && searchSources.length === 0 && llmCalls.length === 0 && !hasKeywords && !hasSearchQuery && !hasBadges) {
            return null;
        }

        const contentWrapper = document.createElement("div");
        
        // Add Search Query
        if (hasSearchQuery) {
            const queryDiv = document.createElement("div");
            queryDiv.style.marginBottom = "12px";
            queryDiv.style.paddingBottom = "12px";
            queryDiv.style.borderBottom = "1px solid var(--border)";
            
            const label = document.createElement("div");
            label.textContent = "搜索语";
            label.style.fontSize = "0.85rem";
            label.style.fontWeight = "600";
            label.style.color = "var(--muted)";
            label.style.marginBottom = "6px";
            
            const text = document.createElement("div");
            text.textContent = searchQuery;
            text.style.fontSize = "0.9rem";
            text.style.color = "var(--text)";
            text.style.lineHeight = "1.5";
            
            queryDiv.appendChild(label);
            queryDiv.appendChild(text);
            contentWrapper.appendChild(queryDiv);
        }

        // Add Keywords and Badges
        if (hasKeywords || hasBadges) {
            const keywordsDiv = document.createElement("div");
            keywordsDiv.style.marginBottom = "12px";
            keywordsDiv.style.paddingBottom = "12px";
            keywordsDiv.style.borderBottom = "1px solid var(--border)";
            
            const label = document.createElement("div");
            label.textContent = "搜索关键词与参数";
            label.style.fontSize = "0.85rem";
            label.style.fontWeight = "600";
            label.style.color = "var(--muted)";
            label.style.marginBottom = "6px";
            
            keywordsDiv.appendChild(label);

            if (hasKeywords) {
                const text = document.createElement("div");
                text.textContent = keywords.join("，");
                text.style.fontSize = "0.9rem";
                text.style.color = "var(--text)";
                text.style.lineHeight = "1.5";
                text.style.marginBottom = hasBadges ? "8px" : "0";
                keywordsDiv.appendChild(text);
            }

            if (hasBadges) {
                keywordsDiv.appendChild(controlFlags);
            }
            
            contentWrapper.appendChild(keywordsDiv);
        }

        // 根据用户选择显示不同的时间信息
        if (hasTotal && state.timingOptions.has('total')) {
            const totalRow = document.createElement("div");
            totalRow.className = "timing-total";
            totalRow.textContent = `总体: ${safeTimings.total_ms.toFixed(2)} ms`;
            contentWrapper.appendChild(totalRow);
        }

        // 添加 Google Vision API 调用标识
        const googleVisionCalled = toolCalls.some(call => call.tool === 'google_vision');
        if (googleVisionCalled && state.timingOptions.has('tools')) {
            const visionBadge = document.createElement("div");
            visionBadge.className = "vision-badge";
            visionBadge.style.marginBottom = "12px";
            visionBadge.style.padding = "6px 12px";
            visionBadge.style.backgroundColor = "#4285f4";
            visionBadge.style.color = "white";
            visionBadge.style.borderRadius = "4px";
            visionBadge.style.fontSize = "0.85rem";
            visionBadge.style.fontWeight = "500";
            visionBadge.style.display = "inline-block";
            
            const visionIcon = document.createElement("span");
            visionIcon.textContent = "👁️ ";
            visionIcon.style.marginRight = "6px";
            
            const visionText = document.createElement("span");
            visionText.textContent = "Google Vision API 已调用";
            
            visionBadge.appendChild(visionIcon);
            visionBadge.appendChild(visionText);
            contentWrapper.appendChild(visionBadge);
        }

        // 创建一个容器用于水平排列搜索源和LLM调用
        const sectionsContainer = document.createElement("div");
        sectionsContainer.className = "timing-sections-container";

        const renderSection = (title, entries, option) => {
            if (!entries.length || !state.timingOptions.has(option)) return null;
            const section = document.createElement("div");
            section.className = "timing-section";
            const sectionTitle = document.createElement("strong");
            sectionTitle.textContent = title;
            section.appendChild(sectionTitle);

            const list = document.createElement("div");
            list.className = "timing-list";

            entries.forEach((entry) => {
                const row = document.createElement("div");
                row.className = "timing-row";

                const label = document.createElement("span");
                label.className = "label";
                label.textContent = entry.label || entry.source || "未知";
                if (entry.error) {
                    label.textContent += `（错误：${entry.error}）`;
                }

                const value = document.createElement("span");
                value.className = "value";
                const duration = Number(entry.duration_ms);
                value.textContent = Number.isFinite(duration) ? `${duration.toFixed(2)} ms` : "--";

                row.appendChild(label);
                row.appendChild(value);
                list.appendChild(row);
            });

            section.appendChild(list);
            return section;
        };

        // 渲染搜索源和LLM调用部分
        const searchSection = renderSection("搜索源", searchSources, 'search');
        const normalizedLLM = llmCalls.map((entry) => {
            const provider = entry.provider || "";
            const model = entry.model || "";
            const suffix = provider && model ? `${provider}/${model}` : provider || model;
            return {
                ...entry,
                label: suffix ? `${entry.label || "LLM"}（${suffix}）` : entry.label || "LLM",
            };
        });
        const llmSection = renderSection("LLM 调用", normalizedLLM, 'llm');

        // 渲染工具调用部分（包括Google Vision API）
        const toolSection = renderSection("工具调用", toolCalls, 'tools');

        // 将搜索源、LLM调用和工具调用添加到水平容器中
        if (searchSection || llmSection || toolSection) {
            contentWrapper.appendChild(sectionsContainer);
            if (searchSection) sectionsContainer.appendChild(searchSection);
            if (llmSection) sectionsContainer.appendChild(llmSection);
            if (toolSection) sectionsContainer.appendChild(toolSection);
        }

        return createCollapsibleSection(
            "响应时间详情", 
            contentWrapper, 
            { 
                sectionClass: "timing-extras",
                collapsed: true
            }
        );
    }

    function buildExtras(data) {
        const fragments = [];

        // 1. 首先添加搜索结果和本地文档
        if (Array.isArray(data.search_hits) && data.search_hits.length > 0) {
            const list = document.createElement("ol");
            list.className = "source-list";
            data.search_hits.forEach((hit, index) => {
                const item = document.createElement("li");
                const title = hit.title || `结果 ${index + 1}`;
                const url = hit.url || "";
                if (url) {
                    const link = document.createElement("a");
                    link.href = url;
                    link.target = "_blank";
                    link.rel = "noopener noreferrer";
                    link.textContent = title;
                    item.appendChild(link);
                } else {
                    const span = document.createElement("span");
                    span.textContent = title;
                    item.appendChild(span);
                }
                if (hit.snippet) {
                    const snippet = document.createElement("div");
                    snippet.textContent = formatter.snippet(hit.snippet, 200);
                    snippet.style.marginTop = "4px";
                    item.appendChild(snippet);
                }
                list.appendChild(item);
            });

            const section = createCollapsibleSection(
                `WEB SOURCES (${data.search_hits.length})`,
                list,
                { bodyClass: "source-scroll" }
            );
            fragments.push(section);
        }

        if (Array.isArray(data.retrieved_docs) && data.retrieved_docs.length > 0) {
            const wrapper = document.createElement("div");
            wrapper.className = "message-extras";

            const heading = document.createElement("h4");
            heading.textContent = "Local Docs";
            wrapper.appendChild(heading);

            const list = document.createElement("ul");
            list.className = "doc-list";
            data.retrieved_docs.forEach((doc, index) => {
                const item = document.createElement("li");
                const source = doc.source || `文档片段 ${index + 1}`;
                const strong = document.createElement("strong");
                strong.textContent = source;
                item.appendChild(strong);

                const snippet = document.createElement("span");
                snippet.textContent = formatter.snippet(doc.content, 240);
                item.appendChild(snippet);

                list.appendChild(item);
            });

            wrapper.appendChild(list);
            fragments.push(wrapper);
        }

        // 2. 然后添加关键词模块 (已移入响应时间详情)
        const control = data.control || {};

        // 3. 最后添加响应时间模块
        // 确保即使没有时间数据，只要有关键词或搜索语，也显示该模块
        const timingExtras = buildTimingExtras(data.response_times, control, data.search_query, data.retrieved_docs);
        if (timingExtras) {
            fragments.push(timingExtras);
        }

        // 4. 添加错误和警告信息
        const metaFragments = [];
        if (data.llm_error) {
            const errorBox = document.createElement("div");
            errorBox.className = "alert-text";
            errorBox.textContent = data.llm_error;
            metaFragments.push(errorBox);
        }
        if (data.llm_warning) {
            const warningBox = document.createElement("div");
            warningBox.className = "warning-text";
            warningBox.textContent = data.llm_warning;
            metaFragments.push(warningBox);
        }
        if (data.search_error) {
            const warningBox = document.createElement("div");
            warningBox.className = "warning-text";
            warningBox.textContent = data.search_error;
            metaFragments.push(warningBox);
        }
        if (data.search_warnings) {
            const warnings = Array.isArray(data.search_warnings) ? data.search_warnings : [data.search_warnings];
            warnings.filter(Boolean).forEach((message) => {
                const warningBox = document.createElement("div");
                warningBox.className = "warning-text";
                warningBox.textContent = message;
                metaFragments.push(warningBox);
            });
        }

        if (metaFragments.length > 0) {
            const wrapper = document.createElement("div");
            wrapper.className = "message-extras";
            metaFragments.forEach(node => wrapper.appendChild(node));
            fragments.push(wrapper);
        }

        if (fragments.length === 0) {
            return null;
        }

        const container = document.createDocumentFragment();
        fragments.forEach(fragment => container.appendChild(fragment));
        return container;
    }

    function setAssistantMessage(messageEl, data) {
        stopLoadingAnimation();
        messageEl.classList.remove("pending");
        const bubble = messageEl.querySelector(".bubble");
        
        // Debug: log the data structure
        console.log("setAssistantMessage data:", data);
        
        // Clear the bubble content
        bubble.innerHTML = '';
        bubble.style.display = 'flex';
        bubble.style.flexDirection = 'column';
        bubble.style.gap = '16px';

        const answer = (data && data.answer) ? String(data.answer).trim() : "";
        if (!answer) {
            console.warn("No answer in response data:", data);
        }
        
        // Create a container for the main answer content
        const contentContainer = document.createElement('div');
        contentContainer.className = 'response-content';
        contentContainer.innerHTML = renderMarkdown(answer || "未能生成答案");
        bubble.appendChild(contentContainer);

        // Create a container for sources and timing info to be displayed after the answer
        const extras = buildExtras(data);

        // Add a separator if both content and extras exist
        if (extras && answer) {
            const separator = document.createElement('hr');
            separator.style.border = 'none';
            separator.style.borderTop = '1px solid var(--border)';
            separator.style.margin = '12px 0';
            bubble.appendChild(separator);
        }

        // Add extras (sources, keywords, timing info) after the main content
        if (extras) {
            bubble.appendChild(extras);
        }

        // Trigger Prism highlight
        if (window.Prism) {
            Prism.highlightAllUnder(bubble);
        }

        // Remove any old extras that might be outside the bubble
        const oldExtras = messageEl.querySelectorAll(".message-extras");
        oldExtras.forEach(node => {
            if (node.parentElement === messageEl) {
                messageEl.removeChild(node);
            }
        });

        scrollToBottom();
    }

    function setAssistantError(messageEl, message) {
        stopLoadingAnimation();
        messageEl.classList.remove("pending");
        const bubble = messageEl.querySelector(".bubble");
        bubble.textContent = message;

        const extras = document.createElement("div");
        extras.className = "message-extras";
        const alert = document.createElement("div");
        alert.className = "alert-text";
        alert.textContent = "请求失败，请检查服务端日志。";
        extras.appendChild(alert);
        messageEl.appendChild(extras);
        scrollToBottom();
    }

    function setLoading(isLoading) {
        state.loading = isLoading;
        sendButton.disabled = isLoading;
        sendButton.classList.toggle("loading", isLoading);
        if (forceSearchButton) {
            forceSearchButton.disabled = isLoading;
        }
        if (isLoading) {
            closeSearchSourceMenu();
        }
        if (editor) {
            editor.setOption("readOnly", isLoading ? "nocursor" : false);
        }
    }

    function updateStatusFromResponse(data) {
        const parts = [];
        if (data.control && data.control.force_search_enabled) {
            parts.push("强制联网：已启用");
        }
        if (data.control && data.control.decision && data.control.decision.reason) {
            parts.push(`路由：${data.control.decision.reason}`);
        }
        if (data.control && data.control.search_mode === "search_unavailable") {
            parts.push("联网搜索不可用，自动切换本地模式");
        }
        if (data.llm_warning) {
            parts.push(`警告：${data.llm_warning}`);
        }
        if (data.llm_error) {
            parts.push(`错误：${data.llm_error}`);
        }
        if (data.search_error) {
            parts.push(`搜索：${data.search_error}`);
        }
        if (data.search_warnings) {
            const warnings = Array.isArray(data.search_warnings) ? data.search_warnings : [data.search_warnings];
            warnings.filter(Boolean).forEach((message) => parts.push(`搜索提示：${message}`));
        }
        statusMessage.textContent = parts.length > 0 ? parts.join(" ｜ ") : "回答已生成";
    }

    function extractCodeBlocks(text) {
        const blocks = [];
        const regex = /```(\w*)\n([\s\S]*?)```/g;
        let match;
        while ((match = regex.exec(text)) !== null) {
            blocks.push({
                lang: match[1] || "text",
                content: match[2]
            });
        }
        return blocks;
    }

    async function handleSubmit(event) {
        event.preventDefault();
        if (state.loading) return;

        const query = editor.getValue().trim();
        if (!query) {
            statusMessage.textContent = "请输入问题。";
            return;
        }

        const userMessage = appendMessage("user", query);
        const assistantMessage = appendMessage("assistant", "");
        // assistantMessage.classList.add("pending"); // Replaced by skeleton loader
        startLoadingAnimation(assistantMessage, searchToggle.checked);

        setLoading(true);
        if (searchToggle.checked) {
            statusMessage.textContent = state.forceSearch
                ? "强制联网：直接进入关键词阶段…"
                : "正在等待搜索结果…";
        } else {
            statusMessage.textContent = "正在等待LLM回应…";
        }
        editor.setValue("");
        // autoResize(); // CodeMirror handles this

        const payload = {
            query,
            search: searchToggle.checked ? "on" : "off",
        };

        const codeBlocks = extractCodeBlocks(query);
        if (codeBlocks.length > 0) {
            payload.code_blocks = codeBlocks;
        }

        if (modelSelect.value) {
            payload.model = modelSelect.value;
        }
        if (searchToggle.checked && state.searchSources.size > 0) {
            payload.search_sources = Array.from(state.searchSources);
        }
        if (searchToggle.checked && state.forceSearch) {
            payload.force_search = true;
        }
        if (state.images.length > 0) {
            payload.images = state.images.map(img => ({
                base64: img.base64,
                mime_type: img.mime_type
            }));
        }
        if (searchToggle.checked) {
            normalizeSearchLimits();
            if (totalLimitInput) {
                const totalValue = parseInt(totalLimitInput.value, 10);
                if (Number.isFinite(totalValue) && totalValue > 0) {
                    payload.search_total_limit = totalValue;
                }
            }
            if (perSourceLimitInput) {
                const perSourceValue = parseInt(perSourceLimitInput.value, 10);
                if (Number.isFinite(perSourceValue) && perSourceValue > 0) {
                    payload.search_source_limit = perSourceValue;
                }
            }
            if (referenceLimitInput) {
                const referenceValue = parseInt(referenceLimitInput.value, 10);
                if (Number.isFinite(referenceValue) && referenceValue > 0) {
                    payload.search_reference_limit = referenceValue;
                }
            }
        }

        try {
            const response = await fetch("/api/answer", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            console.log("Response status:", response.status);
            console.log("Response ok:", response.ok);

            let data;
            try {
                data = await response.json();
                console.log("Parsed JSON data:", data);
            } catch (parseError) {
                console.error("JSON parse error:", parseError);
                throw new Error("响应解析失败");
            }

            if (!response.ok) {
                const message = data?.error || "请求失败";
                console.error("Response not ok:", message);
                statusMessage.textContent = message;
                setAssistantError(assistantMessage, message);
                return;
            }

            // Check if data has the expected structure
            if (!data || typeof data !== 'object') {
                console.error("Invalid response data:", data);
                statusMessage.textContent = "服务器返回无效数据";
                setAssistantError(assistantMessage, "服务器返回数据格式错误");
                return;
            }

            // Check if there's an error in the data
            if (data.error) {
                console.error("Server error:", data.error);
                statusMessage.textContent = data.error;
                setAssistantError(assistantMessage, data.error);
                return;
            }

            if (searchToggle.checked) {
                statusMessage.textContent = "搜索结果已返回，正在等待LLM回应…";
            } else {
                statusMessage.textContent = "正在等待LLM回应…";
            }

            console.log("Setting assistant message with data");
            setAssistantMessage(assistantMessage, data);
            updateStatusFromResponse(data);
        } catch (error) {
            console.error("Request exception:", error);
            console.error("Error stack:", error.stack);
            statusMessage.textContent = "请求失败，请重试";
            setAssistantError(assistantMessage, "抱歉，暂时无法完成请求。");
        } finally {
            setLoading(false);
            editor.focus();
            state.images = [];
            renderImages();
        }
    }

    async function fetchFiles() {
        try {
            const response = await fetch("/api/files");
            if (!response.ok) throw new Error("failed");
            const files = await response.json();
            renderFiles(Array.isArray(files) ? files : []);
        } catch (error) {
            console.error("无法获取文件列表", error);
        }
    }

    function renderFiles(files) {
        fileList.innerHTML = "";
        if (!files || files.length === 0) {
            const chip = document.createElement("div");
            chip.className = "chip empty";
            chip.textContent = "暂未上传文件";
            fileList.appendChild(chip);
            return;
        }

        files.forEach(name => {
            const chip = document.createElement("div");
            chip.className = "chip";
            chip.textContent = name;

            const removeButton = document.createElement("button");
            removeButton.type = "button";
            removeButton.setAttribute("aria-label", `删除 ${name}`);
            removeButton.textContent = "×";
            removeButton.addEventListener("click", async () => {
                try {
                    await fetch(`/api/files/${encodeURIComponent(name)}`, { method: "DELETE" });
                    fetchFiles();
                    statusMessage.textContent = `已删除 ${name}`;
                } catch (error) {
                    console.error(error);
                    statusMessage.textContent = `删除 ${name} 失败`;
                }
            });

            chip.appendChild(removeButton);
            fileList.appendChild(chip);
        });
    }

    async function uploadFiles(files) {
        if (!files || files.length === 0) return;
        
        const imageFiles = [];
        const docFiles = [];
        
        for (const file of files) {
            if (file.type.startsWith('image/')) {
                imageFiles.push(file);
            } else {
                docFiles.push(file);
            }
        }

        if (imageFiles.length > 0) {
            for (const file of imageFiles) {
                if (file.size > 5 * 1024 * 1024) {
                    statusMessage.textContent = `${file.name} 太大 (最大 5MB)`;
                    continue;
                }
                const reader = new FileReader();
                reader.onload = (e) => {
                    state.images.push({
                        name: file.name,
                        base64: e.target.result,
                        mime_type: file.type
                    });
                    renderImages();
                };
                reader.readAsDataURL(file);
            }
            if (docFiles.length === 0) {
                 statusMessage.textContent = `已添加 ${imageFiles.length} 张图片`;
            }
        }

        if (docFiles.length > 0) {
            statusMessage.textContent = "正在上传文件…";

            for (const file of docFiles) {
                const formData = new FormData();
                formData.append("file", file);
                try {
                    const response = await fetch("/api/files", {
                        method: "POST",
                        body: formData,
                    });
                    if (!response.ok) {
                        const error = await response.json().catch(() => ({}));
                        throw new Error(error?.error || "上传失败");
                    }
                } catch (error) {
                    console.error(error);
                    statusMessage.textContent = `${file.name} 上传失败`;
                    return;
                }
            }

            statusMessage.textContent = "文件上传完成";
            await fetchFiles();
        }
    }

    function autoResize() {
        textarea.style.height = "auto";
        textarea.style.height = `${Math.min(textarea.scrollHeight, 320)}px`;
    }

    form.addEventListener("submit", handleSubmit);
    // textarea.addEventListener("input", autoResize);
    // textarea.addEventListener("keydown", (event) => {
    //     if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    //         event.preventDefault();
    //         form.dispatchEvent(new Event("submit", { cancelable: true }));
    //     }
    // });

    uploadButton.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", () => {
        if (fileInput.files && fileInput.files.length > 0) {
            uploadFiles(Array.from(fileInput.files));
            fileInput.value = "";
        }
    });

    if (searchSourceButton) {
        searchSourceButton.addEventListener("click", (event) => {
            event.stopPropagation();
            if (!searchToggle.checked) return;
            toggleSearchSourceMenu();
        });
    }

    if (forceSearchButton) {
        forceSearchButton.addEventListener("click", () => {
            if (!searchToggle.checked || state.loading) return;
            const nextState = !state.forceSearch;
            setForceSearchState(nextState);
            statusMessage.textContent = nextState
                ? "强制联网搜索已开启"
                : "强制联网搜索已关闭";
        });
    }

    for (const checkbox of searchSourceCheckboxes) {
        checkbox.addEventListener("change", handleSearchSourceChange);
    }

    if (timingButton) {
        timingButton.addEventListener("click", (event) => {
            event.stopPropagation();
            if (!timingToggle.checked) return;
            toggleTimingMenu();
        });
    }

    for (const checkbox of timingCheckboxes) {
        checkbox.addEventListener("change", handleTimingChange);
    }

    document.addEventListener("click", (event) => {
        if (!searchSourceDropdown) return;
        if (!searchSourceDropdown.contains(event.target)) {
            closeSearchSourceMenu();
        }
    });

    document.addEventListener("click", (event) => {
        if (!timingDropdown) return;
        if (!timingDropdown.contains(event.target)) {
            closeTimingMenu();
        }
    });

    searchToggle.addEventListener("change", () => {
        updateSearchSourceVisibility();
        if (searchToggle.checked) {
            setForceSearchState(false);
            statusMessage.textContent = "联网搜索已启用";
        } else {
            setForceSearchState(false);
            statusMessage.textContent = "正在等待LLM回应…";
        }
    });

    timingToggle.addEventListener("change", () => {
        updateTimingVisibility();
        if (timingToggle.checked) {
            statusMessage.textContent = "时间详情已启用";
        } else {
            statusMessage.textContent = "时间详情已关闭";
        }
    });

    if (totalLimitInput) {
        totalLimitInput.addEventListener("change", normalizeSearchLimits);
    }

    if (perSourceLimitInput) {
        perSourceLimitInput.addEventListener("change", normalizeSearchLimits);
    }

    if (referenceLimitInput) {
        referenceLimitInput.addEventListener("change", normalizeSearchLimits);
    }

    function renderImages() {
        if (!imagePreviewList) return;
        imagePreviewList.innerHTML = "";
        state.images.forEach((img, index) => {
            const chip = document.createElement("div");
            chip.className = "chip image-chip";
            
            const thumb = document.createElement("img");
            thumb.src = img.base64;
            thumb.style.height = "20px";
            thumb.style.marginRight = "6px";
            thumb.style.verticalAlign = "middle";
            chip.appendChild(thumb);

            const nameSpan = document.createElement("span");
            nameSpan.textContent = img.name;
            chip.appendChild(nameSpan);

            const removeButton = document.createElement("button");
            removeButton.type = "button";
            removeButton.textContent = "×";
            removeButton.addEventListener("click", () => {
                state.images.splice(index, 1);
                renderImages();
            });
            chip.appendChild(removeButton);
            
            imagePreviewList.appendChild(chip);
        });
    }

    ensurePlaceholder();
    fetchFiles();
    loadAvailableModels();
    // autoResize();
    initializeSearchSources();
    initializeTimingOptions();
    updateSearchSourceVisibility();
    updateTimingVisibility();
    normalizeSearchLimits();
    setForceSearchState(false);
});

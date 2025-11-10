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
    const textarea = document.getElementById("query");
    const modelSelect = document.getElementById("model");
    const searchToggle = document.getElementById("search-toggle");
    const searchSourceDropdown = document.getElementById("search-source-dropdown");
    const searchSourceButton = document.getElementById("search-source-button");
    const searchSourceMenu = document.getElementById("search-source-menu");
    const statusMessage = document.getElementById("status-message");
    const sendButton = form.querySelector(".send-btn");
    const uploadButton = document.getElementById("upload-button");
    const fileInput = document.getElementById("file-input");
    const fileList = document.getElementById("file-list");
    const totalLimitInput = document.getElementById("search-total-limit");
    const perSourceLimitInput = document.getElementById("search-per-source-limit");
    const searchSourceCheckboxes = searchSourceMenu
        ? Array.from(searchSourceMenu.querySelectorAll('input[type="checkbox"]'))
        : [];

    const state = {
        loading: false,
        searchSources: new Set(),
    };

    const workflowPanel = document.getElementById("workflow-panel");
    const workflowSearchPreview = document.getElementById("workflow-search-preview");
    const workflowPreviewToggle = document.getElementById("workflow-preview-toggle");
    const workflowStageStatusEls = {
        search: document.getElementById("workflow-stage-search-status"),
        llm: document.getElementById("workflow-stage-llm-status"),
    };
    const workflowStages = {
        search: workflowPanel?.querySelector("[data-stage='search']"),
        llm: workflowPanel?.querySelector("[data-stage='llm']"),
    };
    let workflowPreviewCollapsed = true;
    let workflowPreviewItemCount = 0;
    let collapsibleSectionId = 0;

    function updateWorkflowPreviewToggleLabel(count) {
        if (typeof count === "number") {
            workflowPreviewItemCount = count;
        }
        if (!workflowPreviewToggle) return;
        const suffix = workflowPreviewItemCount > 0 ? ` (${workflowPreviewItemCount})` : "";
        const base = workflowPreviewCollapsed ? "展开全部" : "收起";
        workflowPreviewToggle.textContent = `${base}${suffix}`;
    }

    function setWorkflowPreviewCollapsed(collapsed) {
        workflowPreviewCollapsed = collapsed;
        if (workflowSearchPreview) {
            workflowSearchPreview.classList.toggle("collapsed", collapsed);
            workflowSearchPreview.setAttribute("aria-hidden", collapsed ? "true" : "false");
        }
        if (workflowPreviewToggle) {
            workflowPreviewToggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
        }
        updateWorkflowPreviewToggleLabel();
    }

    function updateWorkflowStage(stage, options = {}) {
        const stageNode = workflowStages[stage];
        if (!stageNode) return;
        stageNode.classList.toggle("active", Boolean(options.active));
        stageNode.classList.toggle("completed", Boolean(options.completed));
        const statusNode = workflowStageStatusEls[stage];
        if (statusNode && typeof options.status === "string") {
            statusNode.textContent = options.status;
        }
    }

    function showWorkflowPreviewNotice(text = "搜索结果将在此展示") {
        if (!workflowSearchPreview) return;
        workflowSearchPreview.innerHTML = "";
        const notice = document.createElement("div");
        notice.className = "workflow-preview-empty";
        notice.textContent = text;
        workflowSearchPreview.appendChild(notice);
        workflowSearchPreview.classList.add("empty");
        updateWorkflowPreviewToggleLabel(0);
        setWorkflowPreviewCollapsed(true);
    }

    function renderWorkflowSearchPreview(data) {
        if (!workflowSearchPreview) return;
        const searchHits = Array.isArray(data?.search_hits) ? data.search_hits : [];
        const docs = Array.isArray(data?.retrieved_docs) ? data.retrieved_docs : [];
        const sections = [];
        const totalCount = searchHits.length + docs.length;

        if (searchHits.length > 0) {
            const list = document.createElement("div");
            list.className = "workflow-preview-list";
            searchHits.slice(0, 4).forEach((hit, index) => {
                const item = document.createElement("div");
                item.className = "workflow-preview-item";

                const title = document.createElement("strong");
                title.textContent = hit.title || hit.url || `搜索结果 ${index + 1}`;
                item.appendChild(title);

                if (hit.url) {
                    const link = document.createElement("a");
                    link.className = "workflow-preview-link";
                    link.href = hit.url;
                    link.target = "_blank";
                    link.rel = "noopener noreferrer";
                    link.textContent = hit.url;
                    item.appendChild(link);
                }

                const snippet = document.createElement("div");
                snippet.className = "workflow-preview-snippet";
                snippet.textContent = formatter.snippet(hit.snippet, 140);
                item.appendChild(snippet);

                list.appendChild(item);
            });

            const group = document.createElement("div");
            group.className = "workflow-preview-group";
            const heading = document.createElement("h4");
            heading.textContent = "搜索条目";
            group.appendChild(heading);
            group.appendChild(list);
            sections.push(group);
        }

        if (docs.length > 0) {
            const list = document.createElement("div");
            list.className = "workflow-preview-list";
            docs.slice(0, 3).forEach((doc, index) => {
                const item = document.createElement("div");
                item.className = "workflow-preview-item";

                const title = document.createElement("strong");
                title.textContent = doc.source || `文档片段 ${index + 1}`;
                item.appendChild(title);

                const snippet = document.createElement("div");
                snippet.className = "workflow-preview-snippet";
                snippet.textContent = formatter.snippet(doc.content || doc.snippet || "", 120);
                item.appendChild(snippet);

                list.appendChild(item);
            });

            const group = document.createElement("div");
            group.className = "workflow-preview-group";
            const heading = document.createElement("h4");
            heading.textContent = "本地文档片段";
            group.appendChild(heading);
            group.appendChild(list);
            sections.push(group);
        }

        if (!sections.length) {
            showWorkflowPreviewNotice("未返回搜索/文档结果");
            return;
        }

        workflowSearchPreview.classList.remove("empty");
        workflowSearchPreview.innerHTML = "";
        sections.forEach((section) => workflowSearchPreview.appendChild(section));
        updateWorkflowPreviewToggleLabel(totalCount);
        setWorkflowPreviewCollapsed(true);
    }

    function startWorkflowSearchStage() {
        updateWorkflowStage("search", {
            active: true,
            completed: false,
            status: "正在等待搜索结果…",
        });
        updateWorkflowStage("llm", {
            active: false,
            completed: false,
            status: "未开始",
        });
        showWorkflowPreviewNotice("搜索结果将在此展示");
    }

    function disableWorkflowSearchStage() {
        updateWorkflowStage("search", {
            active: false,
            completed: false,
            status: "联网搜索未开启",
        });
        updateWorkflowStage("llm", {
            active: true,
            completed: false,
            status: "正在等待LLM回应…",
        });
        showWorkflowPreviewNotice("联网搜索已关闭");
    }

    function markWorkflowSearchResults(data) {
        const searchHits = Array.isArray(data?.search_hits) ? data.search_hits.length : 0;
        const docs = Array.isArray(data?.retrieved_docs) ? data.retrieved_docs.length : 0;
        const total = searchHits + docs;
        updateWorkflowStage("search", {
            active: false,
            completed: true,
            status: total > 0 ? `已获取 ${total} 条结果` : "未找到搜索结果",
        });
        renderWorkflowSearchPreview(data);
    }

    function activateLLMStage(statusText = "正在等待LLM回应…") {
        updateWorkflowStage("llm", {
            active: true,
            completed: false,
            status: statusText,
        });
    }

    function finalizeWorkflowStage() {
        updateWorkflowStage("llm", {
            active: false,
            completed: true,
            status: "LLM 回答已生成",
        });
    }

    function resetWorkflowPanel() {
        updateWorkflowStage("search", {
            active: false,
            completed: false,
            status: "等待触发",
        });
        updateWorkflowStage("llm", {
            active: false,
            completed: false,
            status: "未开始",
        });
        showWorkflowPreviewNotice("搜索结果将在此展示");
    }

    function markWorkflowError(message = "阶段异常") {
        if (searchToggle.checked) {
            updateWorkflowStage("search", {
                active: false,
                completed: true,
                status: "搜索阶段中断",
            });
        }
        updateWorkflowStage("llm", {
            active: false,
            completed: true,
            status: message,
        });
        showWorkflowPreviewNotice("搜索结果不可用");
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
    }

    function normalizeSearchLimits() {
        if (!totalLimitInput || !perSourceLimitInput) return;

        let total = parseInt(totalLimitInput.value, 10);
        if (!Number.isFinite(total) || total < 1) {
            total = 1;
        }
        if (total > 30) {
            total = 30;
        }
        totalLimitInput.value = String(total);

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
        let src = String(md);

        // Handle fenced code blocks first: ```code```
        src = src.replace(/```([\s\S]*?)```/g, (m, code) => {
            const safe = escapeHTML(code);
            return `<pre class="code"><code>${safe}</code></pre>`;
        });

        // Escape remaining HTML to avoid XSS
        src = escapeHTML(src);

        // Inline code
        src = src.replace(/`([^`]+)`/g, (m, code) => `<code>${code}</code>`);
        // Bold and italic (basic)
        src = src.replace(/\*\*([^*]+)\*\*/g, (m, t) => `<strong>${t}</strong>`);
        src = src.replace(/(^|\s)\*([^*]+)\*(?=\s|$)/g, (m, pre, t) => `${pre}<em>${t}</em>`);
        // Links [text](http|https url)
        src = src.replace(/\[([^\]]+)\]\(((?:https?:\/\/)[^\s)]+)\)/g, (m, text, url) =>
            `<a href="${url}" target="_blank" rel="noopener noreferrer">${text}</a>`
        );

        // Headings (#, ##, ###) – keep small visuals using h3
        src = src.replace(/^###\s+(.+)$/gm, (m, t) => `<h3>${t}</h3>`);
        src = src.replace(/^##\s+(.+)$/gm, (m, t) => `<h3>${t}</h3>`);
        src = src.replace(/^#\s+(.+)$/gm, (m, t) => `<h3>${t}</h3>`);

        // Unordered list: lines starting with - or *
        src = src.replace(/^(?:\s*[-*]\s.+(?:\n|$))+?/gm, (block) => {
            const items = block.trim().split(/\n/).map(l => l.replace(/^\s*[-*]\s+/, "").trim());
            const lis = items.map(it => `<li>${it}</li>`).join("");
            return `<ul>${lis}</ul>`;
        });

        // Paragraphs: split by two or more newlines
        const parts = src.split(/\n{2,}/).map(p => {
            // If already a block element, return as is
            if (/^\s*<(?:h3|ul|pre)/.test(p)) return p;
            // Single newlines -> <br>
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

    function buildExtras(data) {
        const fragments = [];

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

        const control = data.control || {};
        const controlFlags = document.createElement("div");
        controlFlags.className = "control-flags";

        if (control.search_allowed === false) {
            controlFlags.appendChild(createBadge("联网：关闭"));
        } else if (control.search_performed) {
            controlFlags.appendChild(createBadge("联网搜索：已触发"));
        } else if (control.search_allowed) {
            controlFlags.appendChild(createBadge("联网搜索：未触发"));
        }

        if (control.local_docs_present) {
            controlFlags.appendChild(createBadge("本地文档：可用"));
        } else if (Array.isArray(data.retrieved_docs) && data.retrieved_docs.length === 0) {
            controlFlags.appendChild(createBadge("本地文档：为空"));
        }

        if (control.hybrid_mode) {
            controlFlags.appendChild(createBadge("混合检索"));
        }

        if (Array.isArray(control.keywords) && control.keywords.length > 0) {
            controlFlags.appendChild(createBadge(`关键词：${control.keywords.join("，")}`));
        }

        if (data.search_query) {
            controlFlags.appendChild(createBadge(`搜索语：${data.search_query}`));
        }

        if (typeof control.search_total_limit === "number") {
            controlFlags.appendChild(createBadge(`汇总上限：${control.search_total_limit}`));
        }

        if (typeof control.search_per_source_limit === "number") {
            controlFlags.appendChild(createBadge(`单源上限：${control.search_per_source_limit}`));
        }

        if (controlFlags.children.length > 0) {
            const wrapper = document.createElement("div");
            wrapper.className = "message-extras";
            wrapper.appendChild(controlFlags);
            fragments.push(wrapper);
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
        
        const extras = buildExtras(data);

        // Create a container for sources
        if (extras) {
            const sourcesContainer = document.createElement('div');
            sourcesContainer.className = 'response-sources';
            sourcesContainer.appendChild(extras);
            const syncSourcesExpansion = () => {
                const hasExpandedSection = Boolean(sourcesContainer.querySelector('.collapsible-section:not(.collapsed)'));
                const hasAlwaysVisible = Boolean(sourcesContainer.querySelector('.message-extras:not(.collapsible-section)'));
                sourcesContainer.classList.toggle('expanded', hasExpandedSection || hasAlwaysVisible);
            };
            sourcesContainer.addEventListener('collapsible-toggle', syncSourcesExpansion);
            syncSourcesExpansion();
            bubble.appendChild(sourcesContainer);
        }

        // Add a separator if both sources and content exist
        if (extras && answer) {
            const separator = document.createElement('hr');
            separator.style.border = 'none';
            separator.style.borderTop = '1px solid var(--border)';
            separator.style.margin = '12px 0';
            bubble.appendChild(separator);
        }

        // Create a container for the main answer content
        const contentContainer = document.createElement('div');
        contentContainer.className = 'response-content';
        contentContainer.innerHTML = renderMarkdown(answer || "未能生成答案");
        bubble.appendChild(contentContainer);

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
        if (isLoading) {
            closeSearchSourceMenu();
        }
    }

    function updateStatusFromResponse(data) {
        const parts = [];
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

    async function handleSubmit(event) {
        event.preventDefault();
        if (state.loading) return;

        const query = textarea.value.trim();
        if (!query) {
            statusMessage.textContent = "请输入问题。";
            return;
        }

        const userMessage = appendMessage("user", query);
        const assistantMessage = appendMessage("assistant", "正在思考…");
        assistantMessage.classList.add("pending");

        setLoading(true);
        if (searchToggle.checked) {
            startWorkflowSearchStage();
            statusMessage.textContent = "正在等待搜索结果…";
        } else {
            disableWorkflowSearchStage();
            statusMessage.textContent = "正在等待LLM回应…";
        }
        textarea.value = "";
        autoResize();

        const payload = {
            query,
            search: searchToggle.checked ? "on" : "off",
        };
        if (modelSelect.value) {
            payload.model = modelSelect.value;
        }
        if (searchToggle.checked && state.searchSources.size > 0) {
            payload.search_sources = Array.from(state.searchSources);
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
                markWorkflowSearchResults(data);
                activateLLMStage("搜索结果已返回，正在等待LLM回应…");
                statusMessage.textContent = "搜索结果已返回，正在等待LLM回应…";
            } else {
                activateLLMStage("LLM 正在生成回答…");
                statusMessage.textContent = "正在等待LLM回应…";
            }

            console.log("Setting assistant message with data");
            setAssistantMessage(assistantMessage, data);
            finalizeWorkflowStage();
            updateStatusFromResponse(data);
        } catch (error) {
            console.error("Request exception:", error);
            console.error("Error stack:", error.stack);
            markWorkflowError("请求阶段异常");
            statusMessage.textContent = "请求失败，请重试";
            setAssistantError(assistantMessage, "抱歉，暂时无法完成请求。");
        } finally {
            setLoading(false);
            textarea.focus();
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
        statusMessage.textContent = "正在上传文件…";

        for (const file of files) {
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

    function autoResize() {
        textarea.style.height = "auto";
        textarea.style.height = `${Math.min(textarea.scrollHeight, 320)}px`;
    }

    form.addEventListener("submit", handleSubmit);
    textarea.addEventListener("input", autoResize);
    textarea.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
            event.preventDefault();
            form.dispatchEvent(new Event("submit", { cancelable: true }));
        }
    });

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

    for (const checkbox of searchSourceCheckboxes) {
        checkbox.addEventListener("change", handleSearchSourceChange);
    }

    document.addEventListener("click", (event) => {
        if (!searchSourceDropdown) return;
        if (!searchSourceDropdown.contains(event.target)) {
            closeSearchSourceMenu();
        }
    });

    searchToggle.addEventListener("change", () => {
        updateSearchSourceVisibility();
        if (searchToggle.checked) {
            resetWorkflowPanel();
            statusMessage.textContent = "联网搜索已启用";
        } else {
            disableWorkflowSearchStage();
            statusMessage.textContent = "联网搜索已关闭";
        }
    });

    if (totalLimitInput) {
        totalLimitInput.addEventListener("change", normalizeSearchLimits);
    }

    if (perSourceLimitInput) {
        perSourceLimitInput.addEventListener("change", normalizeSearchLimits);
    }

    if (workflowPreviewToggle) {
        workflowPreviewToggle.addEventListener("click", () => {
            setWorkflowPreviewCollapsed(!workflowPreviewCollapsed);
        });
    }

    ensurePlaceholder();
    fetchFiles();
    loadAvailableModels();
    autoResize();
    initializeSearchSources();
    updateSearchSourceVisibility();
    normalizeSearchLimits();
    resetWorkflowPanel();
});

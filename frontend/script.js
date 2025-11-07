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
    const statusMessage = document.getElementById("status-message");
    const sendButton = form.querySelector(".send-btn");
    const uploadButton = document.getElementById("upload-button");
    const fileInput = document.getElementById("file-input");
    const fileList = document.getElementById("file-list");

    const state = {
        loading: false,
    };

    async function loadAvailableModels() {
        // Provider display names and sort order
        const providerMeta = {
            glm: { label: "GLM", order: 1 },
            openai: { label: "OpenAI", order: 2 },
            anthropic: { label: "Anthropic", order: 3 },
            google: { label: "Google", order: 4 },
            minimax: { label: "Minimax", order: 5 },
            hkgai: { label: "HKGAI", order: 6 },
            openrouter: { label: "OpenRouter", order: 7 },
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
            defaultOption.textContent = "默认模型 (glm-4.6)";
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
            defaultOption.textContent = "默认模型 (glm-4.6)";
            modelSelect.appendChild(defaultOption);

            const fallback = {
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

            for (const groupName of ["GLM", "OpenAI", "Anthropic", "Google", "Minimax", "HKGAI", "OpenRouter"]) {
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

    function buildExtras(data) {
        const fragments = [];

        if (Array.isArray(data.search_hits) && data.search_hits.length > 0) {
            const wrapper = document.createElement("div");
            wrapper.className = "message-extras";

            const heading = document.createElement("h4");
            heading.textContent = "Web Sources";
            wrapper.appendChild(heading);

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

            wrapper.appendChild(list);
            fragments.push(wrapper);
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
        const answer = (data.answer || "未能生成答案").trim();
        bubble.innerHTML = renderMarkdown(answer);

        const existingExtras = messageEl.querySelectorAll(".message-extras");
        existingExtras.forEach(node => node.remove());

        const extras = buildExtras(data);
        if (extras) {
            messageEl.appendChild(extras);
        }
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
            statusMessage.textContent = "正在生成答案…";
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
        textarea.value = "";
        autoResize();

        const payload = {
            query,
            search: searchToggle.checked ? "on" : "off",
        };
        if (modelSelect.value) {
            payload.model = modelSelect.value;
        }

        try {
            const response = await fetch("/api/answer", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            let data;
            try {
                data = await response.json();
            } catch (parseError) {
                throw new Error("响应解析失败");
            }

            if (!response.ok) {
                const message = data?.error || "请求失败";
                statusMessage.textContent = message;
                setAssistantError(assistantMessage, message);
                return;
            }

            setAssistantMessage(assistantMessage, data);
            updateStatusFromResponse(data);
        } catch (error) {
            console.error(error);
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

    ensurePlaceholder();
    fetchFiles();
    loadAvailableModels();
    autoResize();
});

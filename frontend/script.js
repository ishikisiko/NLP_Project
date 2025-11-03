document.addEventListener("DOMContentLoaded", () => {
    const queryForm = document.getElementById("query-form");
    const uploadForm = document.getElementById("upload-form");
    const fileList = document.getElementById("file-list");
    const statusCard = document.getElementById("status-card");
    const statusMessage = document.getElementById("status-message");
    const answerCard = document.getElementById("answer-card");
    const answerElement = document.getElementById("answer");
    const resultsCard = document.getElementById("results-card");
    const resultsList = document.getElementById("results");
    const retrievedDocsCard = document.getElementById("retrieved-docs-card");
    const retrievedDocsList = document.getElementById("retrieved-docs");

    function showCard(card, visible) {
        card.hidden = !visible;
    }

    async function fetchFiles() {
        const response = await fetch("/api/files");
        const files = await response.json();
        fileList.innerHTML = "";
        files.forEach(file => {
            const li = document.createElement("li");
            li.textContent = file;
            const deleteButton = document.createElement("button");
            deleteButton.textContent = "Delete";
            deleteButton.addEventListener("click", async () => {
                await fetch(`/api/files/${file}`, { method: "DELETE" });
                fetchFiles();
            });
            li.appendChild(deleteButton);
            fileList.appendChild(li);
        });
    }

    uploadForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const formData = new FormData(uploadForm);
        await fetch("/api/files", {
            method: "POST",
            body: formData,
        });
        fetchFiles();
    });

    queryForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const formData = new FormData(queryForm);
        const payload = {
            query: formData.get("query")?.trim(),
            mode: formData.get("mode") || undefined,
            provider: formData.get("provider") || undefined,
            num_results: Number(formData.get("num_results")) || undefined,
            max_tokens: Number(formData.get("max_tokens")) || undefined,
            temperature: Number(formData.get("temperature")) || undefined,
        };

        if (!payload.query) {
            showCard(statusCard, true);
            statusMessage.textContent = "Please enter a question before submitting.";
            return;
        }

        showCard(statusCard, true);
        statusMessage.textContent = "Running search and waiting for the language model...";
        showCard(answerCard, false);
        showCard(resultsCard, false);
        showCard(retrievedDocsCard, false);

        try {
            const response = await fetch("/api/answer", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            const data = await response.json();
            if (!response.ok) {
                const message = data?.error || "Request failed.";
                statusMessage.textContent = message;
                return;
            }

            const {
                answer,
                search_hits: hits = [],
                retrieved_docs: docs = [],
                llm_error,
                llm_warning,
                search_error,
            } = data;
            const metaMessages = [];
            if (llm_error) metaMessages.push(`LLM error: ${llm_error}`);
            if (llm_warning) metaMessages.push(`LLM warning: ${llm_warning}`);
            if (search_error) metaMessages.push(`Search error: ${search_error}`);
            statusMessage.textContent = metaMessages.join(" \u2014 ") || "Done.";

            if (answer) {
                answerElement.textContent = answer;
                showCard(answerCard, true);
            } else {
                showCard(answerCard, false);
            }

            resultsList.innerHTML = "";
            if (hits.length > 0) {
                hits.forEach((hit, index) => {
                    const item = document.createElement("li");
                    const title = hit.title || `Result ${index + 1}`;
                    const link = hit.url || "#";
                    const snippet = hit.snippet || "No snippet available.";
                    item.innerHTML = `
                        <strong><a href="${link}" target="_blank" rel="noopener noreferrer">${title}</a></strong><br>
                        <small>${snippet}</small>
                    `;
                    resultsList.appendChild(item);
                });
                showCard(resultsCard, true);
            } else {
                showCard(resultsCard, false);
            }

            retrievedDocsList.innerHTML = "";
            if (docs.length > 0) {
                docs.forEach((doc, index) => {
                    const item = document.createElement("li");
                    const source = doc.source || `Document ${index + 1}`;
                    const content = doc.content || "No content available.";
                    item.innerHTML = `
                        <strong>${source}</strong><br>
                        <small>${content}</small>
                    `;
                    retrievedDocsList.appendChild(item);
                });
                showCard(retrievedDocsCard, true);
            } else {
                showCard(retrievedDocsCard, false);
            }
        } catch (error) {
            console.error(error);
            statusMessage.textContent = "An unexpected error occurred. Check the server logs for details.";
        }
    });

    fetchFiles();
});

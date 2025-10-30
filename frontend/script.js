document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("query-form");
    const statusCard = document.getElementById("status-card");
    const statusMessage = document.getElementById("status-message");
    const answerCard = document.getElementById("answer-card");
    const answerElement = document.getElementById("answer");
    const resultsCard = document.getElementById("results-card");
    const resultsList = document.getElementById("results");

    function showCard(card, visible) {
        card.hidden = !visible;
    }

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const formData = new FormData(form);
        const payload = {
            query: formData.get("query")?.trim(),
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

            const { answer, search_hits: hits = [], llm_error, llm_warning } = data;
            const metaMessages = [];
            if (llm_error) metaMessages.push(`LLM error: ${llm_error}`);
            if (llm_warning) metaMessages.push(`LLM warning: ${llm_warning}`);
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
        } catch (error) {
            console.error(error);
            statusMessage.textContent = "An unexpected error occurred. Check the server logs for details.";
        }
    });
});

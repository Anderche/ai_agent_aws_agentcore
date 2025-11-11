const chatMessages = document.getElementById("chat-messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const restartButton = document.getElementById("restart-session");
const sessionPill = document.getElementById("chat-session-pill");
const vectorList = document.getElementById("vectorstore-list");
const vectorCount = document.getElementById("vectorstore-count");

let sessionId = null;
let isSending = false;

const createMessageBubble = (text, role) => {
  const bubble = document.createElement("div");
  bubble.classList.add("chat-bubble", role === "user" ? "user" : "agent");
  bubble.textContent = text;
  return bubble;
};

const appendMessage = (text, role = "agent") => {
  const wrapper = document.createElement("div");
  wrapper.className = "flex";
  if (role === "user") {
    wrapper.classList.add("justify-end");
  }
  wrapper.appendChild(createMessageBubble(text, role));
  chatMessages.appendChild(wrapper);
  chatMessages.scrollTop = chatMessages.scrollHeight;
};

const createLoadingMessage = () => {
  const wrapper = document.createElement("div");
  wrapper.className = "flex";
  const bubble = document.createElement("div");
  bubble.className = "chat-bubble agent loading";
  bubble.innerHTML = `
    <span class="loading-copy">Hang tight</span>
    <span class="loading-body">
      Checking latest sources<span class="loading-dots"></span>
    </span>
  `;
  wrapper.appendChild(bubble);
  chatMessages.appendChild(wrapper);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return wrapper;
};

const formatAnswerParagraphs = (text) => {
  if (!text) {
    return ["No answer returned."];
  }

  const trimmed = text.trim();
  if (!trimmed) {
    return ["No answer returned."];
  }

  const manualBlocks = trimmed
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean);
  if (manualBlocks.length > 1) {
    return manualBlocks;
  }

  const sentences = trimmed.match(/[^.!?]+[.!?]?/g) || [trimmed];
  const paragraphs = [];
  let current = [];

  sentences.forEach((sentence) => {
    const normalized = sentence.replace(/\s+/g, " ").trim();
    if (!normalized) {
      return;
    }
    current.push(normalized);
    const exceedsLength = current.join(" ").length >= 220;
    if (current.length >= 2 || exceedsLength) {
      paragraphs.push(current.join(" "));
      current = [];
    }
  });

  if (current.length) {
    paragraphs.push(current.join(" "));
  }

  return paragraphs.length ? paragraphs : [trimmed];
};

const setChatBusy = (state) => {
  isSending = state;
  chatInput.disabled = state;
  chatForm.querySelector("button[type='submit']").disabled = state;
};

const startSession = async () => {
  appendMessage("Connecting to AgentCore assistant…", "agent");
  setChatBusy(true);
  try {
    const response = await fetch("/api/session", { method: "POST" });
    if (!response.ok) {
      throw new Error(`Session error: ${response.statusText}`);
    }
    const data = await response.json();
    sessionId = data.session_id;
    sessionPill.classList.remove("hidden");
    chatMessages.innerHTML = "";
    appendMessage(data.message, "agent");
  } catch (error) {
    console.error(error);
    chatMessages.innerHTML = "";
    appendMessage(
      "Unable to initialize the assistant. Verify the backend server is running and try again.",
      "agent",
    );
  } finally {
    chatInput.value = "";
    setChatBusy(false);
  }
};

const sendChat = async (prompt) => {
  if (!sessionId) {
    await startSession();
  }

  setChatBusy(true);
  appendMessage(prompt, "user");

  const loadingMessage = createLoadingMessage();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, prompt }),
    });

    if (!response.ok) {
      throw new Error(`Chat error: ${response.statusText}`);
    }
    const data = await response.json();
    appendMessage(data.response || "No response received.", "agent");
  } catch (error) {
    console.error(error);
    appendMessage(
      "Something went wrong delivering your message. Please try again.",
      "agent",
    );
  } finally {
    if (loadingMessage && loadingMessage.parentElement) {
      loadingMessage.remove();
    }
    setChatBusy(false);
  }
};

const renderVectorstores = (items) => {
  vectorList.innerHTML = "";
  if (!items.length) {
    vectorList.innerHTML =
      '<div class="rounded-xl border border-dashed border-slate-700/70 bg-slate-900/30 p-8 text-center text-sm text-slate-400">No embedded filings were found. Run the embedding pipeline to populate vectorstores.</div>';
    vectorCount.textContent = "0 available";
    return;
  }

  vectorCount.textContent = `${items.length} embedded`;

  items.forEach((item) => {
    const card = document.createElement("article");
    card.className =
      "space-y-4 rounded-xl border border-slate-800/60 bg-slate-950/60 p-5 shadow-inner shadow-slate-950/40";
    const filingsMeta = [
      item.form ? `Form ${item.form}` : null,
      item.filing_date ? `Filed ${item.filing_date}` : null,
      item.cik ? `CIK ${item.cik}` : null,
    ]
      .filter(Boolean)
      .join(" • ");

    card.innerHTML = `
      <div class="flex flex-col gap-2">
        <div class="flex items-start justify-between gap-3">
          <div>
            <h3 class="text-base font-semibold text-white">${item.label}</h3>
            <p class="text-sm text-slate-400">${filingsMeta || "Metadata not available"}</p>
          </div>
          ${
            item.source_url
              ? `<a href="${item.source_url}" target="_blank" rel="noopener noreferrer" class="inline-flex items-center gap-1 rounded-full bg-slate-800/70 px-3 py-1 text-xs font-medium text-slate-300 hover:text-amber-300">Source<span aria-hidden="true">↗</span></a>`
              : ""
          }
        </div>
        <p class="text-sm text-slate-400">${item.description}</p>
      </div>
      <form class="vector-form space-y-3" data-path="${item.path}">
        <label class="text-xs font-semibold uppercase tracking-wide text-slate-400" for="question-${btoa(
          item.path,
        )}">Ask a question</label>
        <div class="flex flex-col gap-2 sm:flex-row">
          <input
            id="question-${btoa(item.path)}"
            type="text"
            name="question"
            placeholder="What do we need to know from this filing?"
            class="flex-1 rounded-xl border border-slate-700 bg-slate-950/70 px-4 py-2 text-sm text-white placeholder:text-slate-500 focus:border-amber-400 focus:outline-none focus:ring focus:ring-amber-400/30"
            required
          />
          <button
            type="submit"
            class="inline-flex items-center justify-center rounded-xl bg-slate-200 px-4 py-2 text-sm font-semibold text-slate-900 transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-70"
          >
            Run Query
          </button>
        </div>
      </form>
      <div class="vector-answers space-y-3"></div>
    `;

    const form = card.querySelector(".vector-form");
    const input = form.querySelector("input[name='question']");
    const answersContainer = card.querySelector(".vector-answers");

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const question = input.value.trim();
      if (!question) {
        return;
      }

      form.querySelector("button").disabled = true;
      input.disabled = true;

      try {
        const response = await fetch("/api/vectorstores/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            path: form.dataset.path,
            question,
          }),
        });
        if (!response.ok) {
          throw new Error("Vectorstore query failed.");
        }
        const data = await response.json();
        const answerBlock = document.createElement("div");
        answerBlock.className = "vector-answer";
        const answerLabel = document.createElement("div");
        answerLabel.className = "vector-answer-label";
        answerLabel.textContent = "Answer";
        answerBlock.appendChild(answerLabel);

        formatAnswerParagraphs(data.answer).forEach((paragraph) => {
          const p = document.createElement("p");
          p.textContent = paragraph;
          answerBlock.appendChild(p);
        });

        answersContainer.prepend(answerBlock);
        input.value = "";
      } catch (error) {
        console.error(error);
        const errorBlock = document.createElement("div");
        errorBlock.className = "vector-answer";
        errorBlock.textContent =
          "There was an issue querying this filing. Ensure the embedding exists and try again.";
        answersContainer.prepend(errorBlock);
      } finally {
        form.querySelector("button").disabled = false;
        input.disabled = false;
      }
    });

    vectorList.appendChild(card);
  });
};

const loadVectorstores = async () => {
  try {
    const response = await fetch("/api/vectorstores");
    if (!response.ok) {
      throw new Error("Unable to load vectorstores.");
    }
    const data = await response.json();
    renderVectorstores(data.items || []);
  } catch (error) {
    console.error(error);
    vectorCount.textContent = "";
    vectorList.innerHTML =
      '<div class="rounded-xl border border-red-500/30 bg-red-500/10 p-6 text-sm text-red-200">Failed to load embedded filings. Confirm the server can access the vectorstore directory.</div>';
  }
};

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (isSending) return;

  const prompt = chatInput.value.trim();
  if (!prompt) return;

  chatInput.value = "";
  await sendChat(prompt);
});

chatInput.addEventListener("keydown", (event) => {
  if (
    event.key === "Enter" &&
    !event.shiftKey &&
    !event.ctrlKey &&
    !event.metaKey &&
    !event.altKey &&
    !event.isComposing
  ) {
    event.preventDefault();
    if (!isSending) {
      chatForm.requestSubmit();
    }
  }
});

restartButton.addEventListener("click", async () => {
  chatMessages.innerHTML = "";
  appendMessage("Starting a new assistant session…", "agent");
  await startSession();
});

window.addEventListener("DOMContentLoaded", async () => {
  await Promise.all([startSession(), loadVectorstores()]);
});


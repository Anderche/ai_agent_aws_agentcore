import { COMPANIES } from "./companies.js";

const chatMessages = document.getElementById("chat-messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const restartButton = document.getElementById("restart-session");
const sessionPill = document.getElementById("chat-session-pill");
const vectorList = document.getElementById("vectorstore-list");
const vectorCount = document.getElementById("vectorstore-count");
const themeToggle = document.getElementById("theme-toggle");
const infoToggle = document.getElementById("info-toggle");
const infoPanel = document.getElementById("info-panel");
const pickerToggle = document.getElementById("company-picker-toggle");
const picker = document.getElementById("company-picker");
const pickerSearch = document.getElementById("company-search");
const pickerList = document.getElementById("company-list");
const pickerCount = document.getElementById("company-picker-count");

const THEME_STORAGE_KEY = "agentcore-theme";

let sessionId = null;
let isSending = false;
let persistedSymbol = null;

const REFRESH_COMMAND = "/refresh";
const UPPER_SYMBOL_PATTERN = /\b([A-Z]{2,5})\b/g;
const ALPHA_SYMBOL_PATTERN = /\b([A-Za-z]{2,5})\b/g;
const MENU_INTEGER_PATTERN = /^\s*\d{1,3}\.?\s*$/;
const FILINGS_ACTION_PATTERN =
  /(look\s+up\s+(?:sec\s+)?filings?|(?:look\s+up\s+)?sec\s+filings?)/i;
const SYMBOL_STOPWORDS = new Set([
  "A", "AM", "AN", "AND", "ARE", "AS", "AT", "BE", "BY", "CAN", "CHAT",
  "DO", "FAQ", "FILE", "FOR", "FORM", "FROM", "GO", "HAVE", "HE", "HELP",
  "HI", "HTTP", "HTTPS", "HTML", "IF", "IN", "INC", "INTO", "IS", "IT",
  "JSON", "JUST", "LOOK", "LTD", "ME", "MORE", "MY", "NEED", "NEW", "NO",
  "NOT", "OF", "OK", "OLD", "ON", "OR", "OVER", "SEC", "SO", "THAN",
  "THAT", "THE", "THIS", "TO", "TYPE", "UP", "US", "WANT", "WE", "WHAT",
  "WHEN", "WILL", "WITH", "YES", "YOU", "YOUR", "ALL", "CORP",
]);

const normalizeSymbol = (candidate) => {
  if (!candidate) return null;
  const normalized = candidate.trim().toUpperCase();
  if (!/^[A-Z]{1,5}$/.test(normalized)) return null;
  if (SYMBOL_STOPWORDS.has(normalized)) return null;
  return normalized;
};

const isMenuInteger = (text) => MENU_INTEGER_PATTERN.test(text || "");
const isFilingsAction = (text) => FILINGS_ACTION_PATTERN.test(text || "");

const extractSymbolFromText = (text) => {
  if (!text) return null;
  const trimmed = text.trim();
  if (!trimmed) return null;
  if (isMenuInteger(trimmed)) return null;

  const lettersOnly = trimmed.replace(/[^A-Za-z]/g, "");
  if (lettersOnly.toUpperCase() === trimmed.replace(/\s+/g, "").toUpperCase()) {
    const whole = normalizeSymbol(lettersOnly);
    if (whole) return whole;
  }

  for (const pattern of [UPPER_SYMBOL_PATTERN, ALPHA_SYMBOL_PATTERN]) {
    pattern.lastIndex = 0;
    for (const match of trimmed.matchAll(pattern)) {
      const normalized = normalizeSymbol(match[1]);
      if (normalized) {
        return normalized;
      }
    }
  }

  return normalizeSymbol(lettersOnly);
};

const isRefreshCommand = (value) => value.trim().toLowerCase() === REFRESH_COMMAND;

const createMessageBubble = (text, role) => {
  const bubble = document.createElement("div");
  bubble.classList.add("chat-bubble", role === "user" ? "user" : "agent");
  bubble.textContent = text;
  return bubble;
};

const scrollToMessageTop = (element) => {
  const offset =
    element.getBoundingClientRect().top - chatMessages.getBoundingClientRect().top;
  const paddingTop = parseFloat(getComputedStyle(chatMessages).paddingTop) || 0;
  chatMessages.scrollTop += offset - paddingTop;
};

const appendMessage = (text, role = "agent") => {
  const wrapper = document.createElement("div");
  wrapper.className = "flex";
  if (role === "user") {
    wrapper.classList.add("justify-end");
  }
  wrapper.appendChild(createMessageBubble(text, role));
  chatMessages.appendChild(wrapper);
  if (role === "user") {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  } else {
    scrollToMessageTop(wrapper);
  }
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

const startSession = async ({ resetSymbol = true } = {}) => {
  if (resetSymbol) {
    persistedSymbol = null;
  }
  sessionId = null;
  chatMessages.innerHTML = "";
  appendMessage("Connecting to the filing assistant…", "agent");
  setChatBusy(true);
  sessionPill.classList.add("hidden");
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

const handleRefreshCommand = async () => {
  if (isSending) return;
  await startSession({ resetSymbol: true });
};

const sendChat = async (prompt) => {
  if (!sessionId) {
    await startSession({ resetSymbol: false });
  }

  if (!isMenuInteger(prompt) && !isFilingsAction(prompt)) {
    const detected = extractSymbolFromText(prompt);
    if (detected) {
      persistedSymbol = detected;
    }
  }

  setChatBusy(true);
  appendMessage(prompt, "user");

  const loadingMessage = createLoadingMessage();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        prompt,
        ...(persistedSymbol ? { symbol: persistedSymbol } : {}),
      }),
    });

    if (!response.ok) {
      throw new Error(`Chat error: ${response.statusText}`);
    }
    const data = await response.json();
    loadingMessage.remove();
    appendMessage(data.response || "No response received.", "agent");
  } catch (error) {
    console.error(error);
    loadingMessage.remove();
    appendMessage(
      "Something went wrong delivering your message. Please try again.",
      "agent",
    );
  } finally {
    setChatBusy(false);
  }
};

const FILING_FORMS = {
  "10-K": ["Annual report", "Audited yearly overview of the business, financial results, and key risks."],
  "10-Q": ["Quarterly report", "Unaudited quarterly financial results and business update."],
  "8-K": ["Current report", "Announcement of a major event, such as earnings, leadership changes, or deals."],
  "6-K": ["Foreign current report", "Major-event or interim update from a non-U.S. company."],
  "20-F": ["Foreign annual report", "Yearly overview of the business and financials from a non-U.S. company."],
  "DEF 14A": ["Proxy statement", "Shareholder meeting agenda, executive pay, and items up for a vote."],
  "S-1": ["IPO registration", "Registration of new shares, typically ahead of an initial public offering."],
  "3": ["Initial insider holdings", "First disclosure of stock held by a new officer, director, or major shareholder."],
  "4": ["Insider trade", "An officer, director, or major shareholder bought or sold company stock."],
  "5": ["Annual insider summary", "Yearly catch-up report of insider transactions not reported earlier."],
  "SC 13D": ["Activist stake", "An investor disclosed owning more than 5% with intent to influence the company."],
  "SC 13G": ["Passive stake", "An investor disclosed owning more than 5% as a passive holder."],
};

const describeForm = (form) => {
  if (!form) return null;
  const normalized = form.trim().toUpperCase();
  const isAmendment = normalized.endsWith("/A");
  const match = FILING_FORMS[isAmendment ? normalized.slice(0, -2) : normalized];
  if (!match) return null;
  const [name, summary] = match;
  return {
    name: isAmendment ? `${name} (amended)` : name,
    summary: isAmendment ? `Correction or update to an earlier filing. ${summary}` : summary,
  };
};

const formatFilingDate = (value) => {
  if (!value) return null;
  const parsed = new Date(`${value}T00:00:00`);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
};

const escapeHtml = (value) =>
  String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

const renderFilingPill = (key, value, { className = "", title = "" } = {}) =>
  `<span class="filing-pill ${className}"${title ? ` title="${escapeHtml(title)}"` : ""}><span class="filing-pill-key">${key}</span>${escapeHtml(value)}</span>`;

const renderVectorstores = (items) => {
  vectorList.innerHTML = "";
  if (!items.length) {
    vectorList.innerHTML =
      '<div class="rounded-md border border-dashed border-slate-700/70 bg-slate-900/30 p-8 text-center text-sm text-slate-400">No embedded filings were found. Run the embedding pipeline to populate vectorstores.</div>';
    vectorCount.textContent = "0 available";
    return;
  }

  vectorCount.textContent = `${items.length} embedded`;

  items.forEach((item) => {
    const card = document.createElement("article");
    card.className =
      "filing-card space-y-4 rounded-md border border-slate-800/60 bg-slate-950/60 p-5";
    const formInfo = describeForm(item.form);
    const filedOn = formatFilingDate(item.filing_date);
    const pills = [
      item.ticker
        ? renderFilingPill("Ticker", item.ticker, {
            className: "ticker",
            title: "Stock ticker symbol",
          })
        : null,
      item.form
        ? renderFilingPill(
            "Form",
            formInfo ? `${item.form} · ${formInfo.name}` : item.form,
            { title: "SEC filing type" },
          )
        : null,
      filedOn
        ? renderFilingPill("Filed", filedOn, { title: "Date the filing was submitted to the SEC" })
        : null,
    ]
      .filter(Boolean)
      .join("");
    const formSummary = formInfo
      ? formInfo.summary
      : item.form
        ? `SEC Form ${escapeHtml(item.form)} filing.`
        : "Filing type not available.";
    const technicalMeta = [item.cik ? `CIK ${item.cik}` : null, item.label]
      .filter(Boolean)
      .map(escapeHtml)
      .join(" · ");

    card.innerHTML = `
      <div class="flex flex-col gap-3">
        <div class="flex items-start justify-between gap-3">
          <div class="min-w-0">
            <h3 class="font-display text-xl text-white">${escapeHtml(item.company_name || item.label)}</h3>
            <p class="mt-1 text-sm text-slate-400">${formSummary}</p>
          </div>
          ${
            item.source_url
              ? `<a href="${item.source_url}" target="_blank" rel="noopener noreferrer" class="inline-flex shrink-0 items-center gap-1 rounded-full bg-slate-800/70 px-3 py-1 text-xs font-medium text-slate-300 hover:text-amber-300" title="Open the original filing on sec.gov">View on SEC<span aria-hidden="true">↗</span></a>`
              : ""
          }
        </div>
        ${pills ? `<div class="flex flex-wrap gap-2">${pills}</div>` : ""}
        <p class="filing-meta" title="SEC company ID and source document name">${technicalMeta}</p>
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
            class="flex-1 rounded-md border border-slate-700 bg-slate-950/70 px-4 py-2 text-sm text-white placeholder:text-slate-500 focus:border-amber-400 focus:outline-none focus:ring focus:ring-amber-400/30"
            required
          />
          <button
            type="submit"
            class="btn-secondary"
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
          let message =
            "There was an issue querying this filing. Ensure the embedding exists and try again.";
          try {
            const errorBody = await response.json();
            if (errorBody && typeof errorBody.error === "string" && errorBody.error.trim()) {
              message = errorBody.error;
            }
          } catch (parseError) {
            console.error(parseError);
          }
          throw new Error(message);
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
          error instanceof Error && error.message
            ? error.message
            : "There was an issue querying this filing. Ensure the embedding exists and try again.";
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
      '<div class="rounded-md border border-red-500/30 bg-red-500/10 p-6 text-sm text-red-200">Failed to load embedded filings. Confirm the server can access the vectorstore directory.</div>';
  }
};

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (isSending) return;

  const prompt = chatInput.value.trim();
  if (!prompt) return;

  if (isRefreshCommand(prompt)) {
    chatInput.value = "";
    await handleRefreshCommand();
    return;
  }

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
  await handleRefreshCommand();
});

const applyTheme = (theme) => {
  const next = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  if (themeToggle) {
    themeToggle.setAttribute("aria-pressed", next === "light" ? "true" : "false");
  }
};

applyTheme(document.documentElement.dataset.theme || "dark");

if (themeToggle) {
  themeToggle.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    try {
      localStorage.setItem(THEME_STORAGE_KEY, next);
    } catch (error) {
      console.error(error);
    }
    applyTheme(next);
  });
}

const setInfoOpen = (open) => {
  if (!infoPanel || !infoToggle) return;
  infoPanel.hidden = !open;
  infoToggle.setAttribute("aria-expanded", open ? "true" : "false");
};

if (infoToggle && infoPanel) {
  infoToggle.addEventListener("click", (event) => {
    event.stopPropagation();
    setInfoOpen(infoPanel.hidden);
  });

  document.addEventListener("click", (event) => {
    if (infoPanel.hidden) return;
    const target = event.target;
    if (target instanceof Node && (infoPanel.contains(target) || infoToggle.contains(target))) {
      return;
    }
    setInfoOpen(false);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      setInfoOpen(false);
    }
  });
}

let pickerMatches = COMPANIES;
let pickerActiveIndex = -1;

const rankCompany = (company, query) => {
  const ticker = company.ticker.toLowerCase();
  const name = company.name.toLowerCase();
  if (ticker === query) return 0;
  if (ticker.startsWith(query)) return 1;
  if (name.startsWith(query)) return 2;
  if (name.split(/[\s\-&.()]+/).some((word) => word.startsWith(query))) return 3;
  if (name.includes(query)) return 4;
  return null;
};

const filterCompanies = (rawQuery) => {
  const query = rawQuery.trim().toLowerCase();
  if (!query) return COMPANIES;
  return COMPANIES.map((company) => ({ company, rank: rankCompany(company, query) }))
    .filter((entry) => entry.rank !== null)
    .sort((a, b) => a.rank - b.rank)
    .map((entry) => entry.company);
};

const setPickerActive = (index) => {
  const options = pickerList.querySelectorAll("[role='option']");
  options.forEach((option, i) => option.setAttribute("aria-selected", i === index ? "true" : "false"));
  pickerActiveIndex = index;
  const active = options[index];
  if (active) {
    pickerSearch.setAttribute("aria-activedescendant", active.id);
    active.scrollIntoView({ block: "nearest" });
  } else {
    pickerSearch.removeAttribute("aria-activedescendant");
  }
};

const renderCompanyList = () => {
  pickerMatches = filterCompanies(pickerSearch.value);
  pickerList.innerHTML = "";
  pickerCount.textContent = pickerSearch.value.trim()
    ? `${pickerMatches.length} of ${COMPANIES.length}`
    : `${COMPANIES.length} companies`;

  if (!pickerMatches.length) {
    const empty = document.createElement("li");
    empty.className = "company-list-empty";
    empty.textContent = "No match in this demo list. Try the SEC search below.";
    pickerList.appendChild(empty);
    setPickerActive(-1);
    return;
  }

  pickerMatches.forEach((company, index) => {
    const option = document.createElement("li");
    option.id = `company-option-${company.ticker}`;
    option.className = "company-option";
    option.setAttribute("role", "option");
    option.setAttribute("aria-selected", "false");
    const name = document.createElement("span");
    name.className = "truncate";
    name.textContent = company.name;
    const ticker = document.createElement("span");
    ticker.className = "company-ticker";
    ticker.textContent = company.ticker;
    option.append(name, ticker);
    option.addEventListener("mousedown", (event) => event.preventDefault());
    option.addEventListener("click", () => selectCompany(company));
    option.addEventListener("mousemove", () => {
      if (pickerActiveIndex !== index) setPickerActive(index);
    });
    pickerList.appendChild(option);
  });

  setPickerActive(pickerSearch.value.trim() ? 0 : -1);
};

const fitPickerToPanel = () => {
  const panel = chatForm.closest("section");
  if (!panel) return;
  const available = chatForm.getBoundingClientRect().top - panel.getBoundingClientRect().top;
  picker.style.maxHeight = `${Math.max(available, 160)}px`;
};

const setPickerOpen = (open) => {
  picker.hidden = !open;
  pickerToggle.setAttribute("aria-expanded", open ? "true" : "false");
  if (open) {
    fitPickerToPanel();
    pickerSearch.value = "";
    renderCompanyList();
    pickerList.scrollTop = 0;
    pickerSearch.focus();
  }
};

const selectCompany = (company) => {
  persistedSymbol = company.ticker;
  const entry = `${company.ticker} (${company.name})`;
  const current = chatInput.value.trimEnd();
  chatInput.value = current ? `${current} ${entry}` : entry;
  setPickerOpen(false);
  chatInput.focus();
  chatInput.setSelectionRange(chatInput.value.length, chatInput.value.length);
};

pickerToggle.addEventListener("click", (event) => {
  event.stopPropagation();
  setPickerOpen(picker.hidden);
});

pickerSearch.addEventListener("input", renderCompanyList);

window.addEventListener("resize", () => {
  if (!picker.hidden) fitPickerToPanel();
});

pickerSearch.addEventListener("keydown", (event) => {
  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    if (!pickerMatches.length) return;
    const step = event.key === "ArrowDown" ? 1 : -1;
    const next =
      pickerActiveIndex === -1
        ? step === 1
          ? 0
          : pickerMatches.length - 1
        : (pickerActiveIndex + step + pickerMatches.length) % pickerMatches.length;
    setPickerActive(next);
  } else if (event.key === "Enter") {
    event.preventDefault();
    const company = pickerMatches[pickerActiveIndex];
    if (company) selectCompany(company);
  } else if (event.key === "Escape") {
    event.stopPropagation();
    setPickerOpen(false);
    pickerToggle.focus();
  }
});

document.addEventListener("click", (event) => {
  if (picker.hidden) return;
  const target = event.target;
  if (target instanceof Node && (picker.contains(target) || pickerToggle.contains(target))) {
    return;
  }
  setPickerOpen(false);
});

document.querySelectorAll("[data-collapse-target]").forEach((toggle) => {
  const target = document.getElementById(toggle.dataset.collapseTarget);
  if (!target) return;
  toggle.addEventListener("click", () => {
    const open = toggle.getAttribute("aria-expanded") !== "true";
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
    target.dataset.collapsed = open ? "false" : "true";
  });
});

window.addEventListener("DOMContentLoaded", async () => {
  await Promise.all([startSession(), loadVectorstores()]);
});


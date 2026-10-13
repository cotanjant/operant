// Operant Interactive Playground & Docs Script

document.addEventListener('DOMContentLoaded', () => {
  // Tab switching logic
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-target');

      tabBtns.forEach(b => {
        b.classList.remove('bg-brand-500/20', 'text-brand-400', 'font-semibold');
        b.classList.add('text-slate-400');
      });
      btn.classList.add('bg-brand-500/20', 'text-brand-400', 'font-semibold');
      btn.classList.remove('text-slate-400');

      tabContents.forEach(content => {
        if (content.id === targetId) {
          content.classList.remove('hidden');
        } else {
          content.classList.add('hidden');
        }
      });
    });
  });

  // API Status Check
  const statusDot = document.getElementById('api-status-dot');
  const statusText = document.getElementById('api-status-text');
  const endpointInput = document.getElementById('api-endpoint');
  const recheckBtn = document.getElementById('btn-recheck-api');

  let isOnline = false;

  async function checkApiHealth() {
    const base = endpointInput.value.trim().replace(/\/$/, '');
    statusDot.className = 'w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse';
    statusText.textContent = `Checking ${base}...`;

    try {
      const res = await fetch(`${base}/health`, { method: 'GET', mode: 'cors' });
      if (res.ok) {
        isOnline = true;
        statusDot.className = 'w-2.5 h-2.5 rounded-full bg-brand-400';
        statusText.textContent = 'API Live at localhost:8000';
      } else {
        throw new Error('Non-200 response');
      }
    } catch (err) {
      isOnline = false;
      statusDot.className = 'w-2.5 h-2.5 rounded-full bg-slate-500';
      statusText.textContent = 'Offline (Simulated agentic mode active)';
    }
  }

  checkApiHealth();
  if (recheckBtn) recheckBtn.addEventListener('click', checkApiHealth);

  // Chat playground (multi-turn: full history sent each request)
  const chatMessages = document.getElementById('chat-messages');
  const chatForm = document.getElementById('chat-form');
  const chatInput = document.getElementById('chat-input');
  const tempInput = document.getElementById('api-temp');
  const clearChatBtn = document.getElementById('clear-chat');
  let history = [];

  function appendMessage(role, text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'flex items-start gap-3';

    const isUser = role === 'user';
    const avatar = isUser
      ? `<div class="w-8 h-8 rounded-lg bg-accent-500/20 text-accent-400 border border-accent-500/30 flex items-center justify-center font-bold text-sm shrink-0">U</div>`
      : `<div class="w-8 h-8 rounded-lg bg-brand-500/20 text-brand-400 border border-brand-500/30 flex items-center justify-center font-bold text-sm shrink-0">O</div>`;

    const bubbleStyle = isUser
      ? 'bg-slate-800 border border-slate-700 rounded-2xl rounded-tr-sm p-4 text-sm text-slate-100 max-w-2xl ml-auto'
      : 'bg-slate-900 border border-slate-800 rounded-2xl rounded-tl-sm p-4 text-sm text-slate-200 max-w-2xl leading-relaxed whitespace-pre-wrap';

    msgDiv.innerHTML = isUser
      ? `<div class="${bubbleStyle}">${escapeHtml(text)}</div>${avatar}`
      : `${avatar}<div class="${bubbleStyle}">${escapeHtml(text)}</div>`;

    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return msgDiv;
  }

  function escapeHtml(string) {
    const div = document.createElement('div');
    div.innerText = string;
    return div.innerHTML;
  }

  if (clearChatBtn) {
    clearChatBtn.addEventListener('click', () => {
      chatMessages.innerHTML = '';
      history = [];
      appendMessage('assistant', 'Hello! I am Operant, an agentic native model fine-tuned for planning, tool execution, and code synthesis. How can I assist you today?');
    });
  }

  // Quick prompt buttons
  document.querySelectorAll('.quick-prompt').forEach(btn => {
    btn.addEventListener('click', () => {
      const prompt = btn.getAttribute('data-prompt');
      if (prompt && chatInput) {
        chatInput.value = prompt;
        chatForm.dispatchEvent(new Event('submit'));
      }
    });
  });

  // Handle Form Submission
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = chatInput.value.trim();
    if (!message) return;

    appendMessage('user', message);
    history.push({ role: 'user', content: message });
    chatInput.value = '';

    const loadingDiv = appendMessage('assistant', 'Thinking...');
    const bubble = loadingDiv.querySelector('.bg-slate-900');

    const base = endpointInput.value.trim().replace(/\/$/, '');
    const temperature = parseFloat(tempInput.value) || 0.7;

    try {
      if (isOnline) {
        const response = await fetch(`${base}/v1/chat/completions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: 'operant',
            messages: history,
            temperature: temperature,
            max_tokens: 512,
            stream: false
          })
        });

        if (!response.ok) {
          throw new Error(`Server returned HTTP ${response.status}`);
        }

        const data = await response.json();
        const reply = data.choices[0]?.message?.content || 'No response returned.';
        bubble.innerText = reply;
        history.push({ role: 'assistant', content: reply });
      } else {
        // Fallback demo simulation
        await new Promise(r => setTimeout(r, 600));
        let simulatedReply = '';

        if (message.toLowerCase().includes('architecture')) {
          simulatedReply = 'Operant is built on Gemma-3 (270M parameters) with hybrid sliding-window attention (512 tokens window) and full attention every 6th layer. Inference is handled natively by Candle in pure Rust, exposing an OpenAI-compatible API on :8000.';
        } else if (message.toLowerCase().includes('plan') || message.toLowerCase().includes('bug')) {
          simulatedReply = '1. Reproduce: Run `cargo test` to capture assertion failure.\n2. Locate: Grep caller functions across the codebase.\n3. Root Cause: Check boundary conditions and invalid invariants.\n4. Patch: Apply minimal contiguous change in shared code.\n5. Verify: Run test suite to verify 0 regressions.';
        } else if (message.toLowerCase().includes('who are you') || message.toLowerCase().includes('who am i')) {
          simulatedReply = 'I am Operant, an agentic native model fine-tuned for planning, tool execution, and code synthesis with a high-performance pure-Rust inference engine.';
        } else {
          simulatedReply = `[Simulated Operant Agent response for "${message}"]: Operant is ready. To run local native inference with pure Rust, execute: cargo run -- serve --model-file ../model/out`;
        }

        bubble.innerText = simulatedReply;
        history.push({ role: 'assistant', content: simulatedReply });
      }
    } catch (err) {
      bubble.innerText = `Error connecting to API: ${err.message}. Ensure the Operant server is running via 'cargo run -- serve'.`;
    }

    chatMessages.scrollTop = chatMessages.scrollHeight;
  });
});

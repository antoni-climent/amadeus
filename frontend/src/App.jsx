import { useEffect, useRef, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";
const STREAMING_ASSISTANT_ID = "streaming-assistant";
const PORTRAIT_FRAMES = {
  neutral: [
    "/kurisu/neutral/neutral-closed.png",
    "/kurisu/neutral/neutral-mid.png",
    "/kurisu/neutral/neutral-open.png",
  ],
  thinking: [
    "/kurisu/thinking/thinking-closed.png",
    "/kurisu/thinking/thinking-mid.png",
    "/kurisu/thinking/thinking-open.png",
  ],
  annoyed: [
    "/kurisu/annoyed/annoyed-closed.png",
    "/kurisu/annoyed/annoyed-mid.png",
    "/kurisu/annoyed/annoyed-open.png",
  ],
  happy: [
    "/kurisu/happy/happy-closed.png",
    "/kurisu/happy/happy-mid.png",
    "/kurisu/happy/happy-open.png",
  ],
  surprised: [
    "/kurisu/surprised/surprised-closed.png",
    "/kurisu/surprised/surprised-mid.png",
    "/kurisu/surprised/surprised-open.png",
  ],
  sad: [
    "/kurisu/sad/sad-closed.png",
    "/kurisu/sad/sad-mid.png",
    "/kurisu/sad/sad-open.png",
  ],
};
const EXPRESSION_LABELS = {
  neutral: "Neutral",
  thinking: "Thinking",
  annoyed: "Annoyed",
  happy: "Happy",
  surprised: "Surprised",
  sad: "Sad",
};

function toDisplayMessages(history, streamingResponse) {
  const messages = history.map((entry, index) => ({
    id: `${entry.role}-${index}`,
    role: entry.role,
    content: entry.content,
  }));

  if (streamingResponse) {
    messages.push({
      id: STREAMING_ASSISTANT_ID,
      role: "assistant",
      content: streamingResponse,
    });
  }

  return messages;
}

function detectExpression(text, isWaitingForStream) {
  if (isWaitingForStream) {
    return "thinking";
  }

  const normalized = text.toLowerCase();
  if (!normalized.trim()) {
    return "neutral";
  }

  if (
    normalized.includes("idiot") ||
    normalized.includes("pervert") ||
    normalized.includes("honestly") ||
    normalized.includes("nonsense") ||
    normalized.includes("ridiculous")
  ) {
    return "annoyed";
  }

  if (
    normalized.includes("sorry") ||
    normalized.includes("sad") ||
    normalized.includes("unfortunately") ||
    normalized.includes("that's rough") ||
    normalized.includes("i understand")
  ) {
    return "sad";
  }

  if (
    normalized.includes("wait") ||
    normalized.includes("what?") ||
    normalized.includes("impossible") ||
    normalized.includes("seriously?") ||
    normalized.includes("that's impossible")
  ) {
    return "surprised";
  }

  if (
    normalized.includes("glad") ||
    normalized.includes("thank you") ||
    normalized.includes("good") ||
    normalized.includes("nice") ||
    normalized.includes("happy")
  ) {
    return "happy";
  }

  if (
    normalized.includes("technically") ||
    normalized.includes("let me think") ||
    normalized.includes("in theory") ||
    normalized.includes("actually") ||
    normalized.includes("first")
  ) {
    return "thinking";
  }

  return "neutral";
}

export default function App() {
  const [history, setHistory] = useState([]);
  const [message, setMessage] = useState("");
  const [streamingResponse, setStreamingResponse] = useState("");
  const [status, setStatus] = useState("Checking backend...");
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [isWaitingForStream, setIsWaitingForStream] = useState(false);
  const [mouthFrame, setMouthFrame] = useState(0);
  const [error, setError] = useState("");
  const messagesEndRef = useRef(null);
  const mouthTimeoutsRef = useRef([]);
  const previousStreamTextRef = useRef("");
  const displayedMessages = toDisplayMessages(history, streamingResponse);
  const latestAssistantText =
    streamingResponse ||
    [...history].reverse().find((entry) => entry.role === "assistant")?.content ||
    "";
  const expression = detectExpression(latestAssistantText, isWaitingForStream);
  const portraitFrames = PORTRAIT_FRAMES[expression] ?? PORTRAIT_FRAMES.neutral;
  const portraitSource = portraitFrames[mouthFrame] ?? portraitFrames[0];
  const isSpeaking = isSending && !isWaitingForStream;

  useEffect(() => {
    checkHealth();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history, streamingResponse, isSending]);

  useEffect(() => {
    if (!isSpeaking) {
      previousStreamTextRef.current = "";
      clearMouthAnimation(mouthTimeoutsRef);
      setMouthFrame(0);
      return undefined;
    }

    const previousText = previousStreamTextRef.current;
    const deltaText = streamingResponse.startsWith(previousText)
      ? streamingResponse.slice(previousText.length)
      : streamingResponse;

    previousStreamTextRef.current = streamingResponse;

    if (!deltaText.trim()) {
      return undefined;
    }

    clearMouthAnimation(mouthTimeoutsRef);

    const lastMeaningfulChar = deltaText.trim().slice(-1);
    const isPause = /[.,!?;:]/.test(lastMeaningfulChar);
    const sequence = chooseMouthSequence(deltaText, isPause);
    const frameDuration = isPause ? 90 : 70;

    sequence.forEach((frame, index) => {
      const timeoutId = window.setTimeout(() => {
        setMouthFrame(frame);
      }, frameDuration * index);
      mouthTimeoutsRef.current.push(timeoutId);
    });

    return () => undefined;
  }, [isSpeaking, streamingResponse]);

  useEffect(() => {
    return () => clearMouthAnimation(mouthTimeoutsRef);
  }, []);

  async function checkHealth() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (!response.ok) {
        throw new Error("Backend is unavailable.");
      }
      const data = await response.json();
      setStatus(data.model_loaded ? "Backend ready." : "Backend up. Model not loaded.");
    } catch {
      setStatus("Backend unreachable.");
    }
  }

  async function loadModel() {
    if (isSending) {
      return;
    }

    setIsLoadingModel(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE_URL}/load`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail ?? "Failed to load model.");
      }
      setStatus("Model loaded.");
    } catch (loadError) {
      setError(loadError.message);
    } finally {
      setIsLoadingModel(false);
    }
  }

  async function sendMessage(event) {
    event?.preventDefault();
    const trimmedMessage = message.trim();
    if (!trimmedMessage || isSending) {
      return;
    }

    setIsSending(true);
    setIsWaitingForStream(true);
    setError("");
    setStreamingResponse("");
    setStatus("Sending message...");
    const nextHistory = [...history, { role: "user", content: trimmedMessage }];
    setHistory(nextHistory);
    setMessage("");

    try {
      const response = await fetch(`${API_BASE_URL}/generate/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: trimmedMessage,
          history,
        }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail ?? "Failed to generate response.");
      }

      if (!response.body) {
        throw new Error("Streaming is not available.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";

        for (const eventBlock of events) {
          const parsedEvent = parseSseEvent(eventBlock);
          if (!parsedEvent) {
            continue;
          }

          if (parsedEvent.event === "delta") {
            setIsWaitingForStream(false);
            setStatus("Kurisu is replying...");
            setStreamingResponse(parsedEvent.data.response ?? "");
          }

          if (parsedEvent.event === "done") {
            setStreamingResponse("");
            setIsWaitingForStream(false);
            setHistory(parsedEvent.data.history ?? []);
            setStatus("Backend ready.");
          }
        }
      }
    } catch (sendError) {
      setHistory(history);
      setError(sendError.message);
      setStreamingResponse("");
      setIsWaitingForStream(false);
      setStatus("Backend ready.");
    } finally {
      setIsSending(false);
    }
  }

  function handleComposerKeyDown(event) {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      sendMessage(event);
    }
  }

  return (
    <main className="app-shell">
      <section className={`scene expression-${expression}`}>
        <div className="scene-overlay" />
        <div className="character-stage">
          <div className="character-glow" />
          <img
            className="character-image"
            src={portraitSource}
            alt={`Makise Kurisu ${expression}`}
          />
        </div>

        <section className="chat-overlay">
          <header className="chat-header">
            <div>
              <p className="eyebrow">Amadeus</p>
              <h1>Kurisu Chat</h1>
              <p className="status-text">{status}</p>
            </div>
            <div className="chat-meta">
              <span className="expression-chip">{EXPRESSION_LABELS[expression]}</span>
              <span className="portrait-status">
                {isWaitingForStream ? "Preparing" : isSpeaking ? "Speaking" : "Idle"}
              </span>
            </div>
            <button className="secondary-button" onClick={loadModel} disabled={isLoadingModel || isSending}>
              {isLoadingModel ? "Loading..." : "Load model"}
            </button>
          </header>

          <section className="messages">
            {displayedMessages.length === 0 ? (
              <div className="empty-state">
                <p>Start the conversation.</p>
                <p>The app keeps the returned history and streams Kurisu&apos;s reply live.</p>
              </div>
            ) : (
              displayedMessages.map((entry) => (
                <article
                  key={entry.id}
                  className={`message ${entry.role}${entry.id === STREAMING_ASSISTANT_ID ? " pending" : ""}`}
                >
                  <p className="message-role">{entry.role === "assistant" ? "Kurisu" : "You"}</p>
                  <p className="message-content">{entry.content}</p>
                </article>
              ))
            )}
            {isSending && isWaitingForStream ? (
              <article className="message assistant pending typing-message">
                <p className="message-role">Kurisu</p>
                <div className="typing-indicator" aria-label="Kurisu is typing">
                  <span />
                  <span />
                  <span />
                </div>
              </article>
            ) : null}
            <div ref={messagesEndRef} />
          </section>

          <form className="composer" onSubmit={sendMessage}>
          <textarea
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            onKeyDown={handleComposerKeyDown}
            placeholder={isSending ? "Wait for the current reply to finish..." : "Send a message..."}
            rows={3}
            disabled={isSending || isLoadingModel}
          />
            <button className="primary-button" type="submit" disabled={isSending || !message.trim()}>
              {isSending ? "Streaming..." : "Send"}
            </button>
          </form>

          {error ? <p className="error-banner">{error}</p> : null}
        </section>
      </section>
    </main>
  );
}

function parseSseEvent(eventBlock) {
  const lines = eventBlock.split("\n");
  let eventName = "";
  let data = "";

  for (const line of lines) {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    }
    if (line.startsWith("data:")) {
      data += line.slice(5).trim();
    }
  }

  if (!eventName || !data) {
    return null;
  }

  return { event: eventName, data: JSON.parse(data) };
}

function chooseMouthSequence(deltaText, isPause) {
  if (isPause) {
    return [1, 0];
  }

  const compactText = deltaText.replace(/\s+/g, "");
  if (compactText.length <= 2) {
    return [1, 0];
  }

  if (compactText.length <= 6) {
    return [1, 2, 0];
  }

  return [1, 2, 1, 2, 0];
}

function clearMouthAnimation(mouthTimeoutsRef) {
  for (const timeoutId of mouthTimeoutsRef.current) {
    window.clearTimeout(timeoutId);
  }
  mouthTimeoutsRef.current = [];
}

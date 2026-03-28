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
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [error, setError] = useState("");
  const messagesEndRef = useRef(null);
  const audioRef = useRef(null);
  const audioObjectUrlRef = useRef("");
  const hasAttemptedAutoLoadRef = useRef(false);
  const displayedMessages = toDisplayMessages(history, streamingResponse);
  const latestAssistantText =
    streamingResponse ||
    [...history].reverse().find((entry) => entry.role === "assistant")?.content ||
    "";
  const expression = detectExpression(latestAssistantText, isWaitingForStream);
  const portraitFrames = PORTRAIT_FRAMES[expression] ?? PORTRAIT_FRAMES.neutral;
  const portraitSource = portraitFrames[mouthFrame] ?? portraitFrames[0];
  const portraitStatus = isWaitingForStream ? "Preparing" : isPlayingAudio ? "Voice" : "Idle";

  useEffect(() => {
    checkHealth();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history, streamingResponse, isSending]);

  useEffect(() => {
    if (!isPlayingAudio) {
      setMouthFrame(0);
      return undefined;
    }

    const intervalId = window.setInterval(() => {
      setMouthFrame((currentFrame) => (currentFrame + 1) % 3);
    }, 130);

    return () => window.clearInterval(intervalId);
  }, [isPlayingAudio]);

  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
      }
      if (audioObjectUrlRef.current) {
        URL.revokeObjectURL(audioObjectUrlRef.current);
      }
    };
  }, []);

  async function checkHealth() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (!response.ok) {
        throw new Error("Backend is unavailable.");
      }
      const data = await response.json();
      if (data.model_loaded) {
        setStatus("Backend ready.");
        return;
      }

      setStatus("Backend up. Loading model...");
      if (!hasAttemptedAutoLoadRef.current) {
        hasAttemptedAutoLoadRef.current = true;
        void loadModel();
      }
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
            await playSpeech(parsedEvent.data.response ?? "");
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

  async function playSpeech(text) {
    if (!text.trim()) {
      return;
    }

    try {
      setStatus("Synthesizing voice...");
      const response = await fetch(`${API_BASE_URL}/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: String(text) }),
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.detail ?? "Failed to synthesize speech.");
      }

      const audioBlob = await response.blob();
      if (audioObjectUrlRef.current) {
        URL.revokeObjectURL(audioObjectUrlRef.current);
      }

      const objectUrl = URL.createObjectURL(audioBlob);
      audioObjectUrlRef.current = objectUrl;
      const audio = new Audio(objectUrl);
      audioRef.current = audio;

      audio.onplay = () => {
        setIsPlayingAudio(true);
        setStatus("Playing voice...");
      };
      audio.onended = () => {
        setIsPlayingAudio(false);
        setMouthFrame(0);
        setStatus("Backend ready.");
      };
      audio.onerror = () => {
        setIsPlayingAudio(false);
        setMouthFrame(0);
        setError("Audio playback failed.");
        setStatus("Backend ready.");
      };

      await audio.play();
    } catch (speechError) {
      setIsPlayingAudio(false);
      setMouthFrame(0);
      setError(speechError.message);
      setStatus("Backend ready.");
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
              <span className="portrait-status">{portraitStatus}</span>
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

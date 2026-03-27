import { useEffect, useRef, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";
const STREAMING_ASSISTANT_ID = "streaming-assistant";

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

export default function App() {
  const [history, setHistory] = useState([]);
  const [message, setMessage] = useState("");
  const [streamingResponse, setStreamingResponse] = useState("");
  const [status, setStatus] = useState("Checking backend...");
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState("");
  const messagesEndRef = useRef(null);
  const displayedMessages = toDisplayMessages(history, streamingResponse);

  useEffect(() => {
    checkHealth();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history, streamingResponse, isSending]);

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
    event.preventDefault();
    const trimmedMessage = message.trim();
    if (!trimmedMessage || isSending) {
      return;
    }

    setIsSending(true);
    setError("");
    setStreamingResponse("");
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
            setStreamingResponse(parsedEvent.data.response ?? "");
          }

          if (parsedEvent.event === "done") {
            setStreamingResponse("");
            setHistory(parsedEvent.data.history ?? []);
            setStatus("Backend ready.");
          }
        }
      }
    } catch (sendError) {
      setHistory(history);
      setError(sendError.message);
      setStreamingResponse("");
    } finally {
      setIsSending(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="chat-panel">
        <header className="chat-header">
          <div>
            <p className="eyebrow">Amadeus</p>
            <h1>Kurisu Chat</h1>
            <p className="status-text">{status}</p>
          </div>
          <button className="secondary-button" onClick={loadModel} disabled={isLoadingModel}>
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
          <div ref={messagesEndRef} />
        </section>

        <form className="composer" onSubmit={sendMessage}>
          <textarea
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            placeholder="Send a message..."
            rows={3}
          />
          <button className="primary-button" type="submit" disabled={isSending || !message.trim()}>
            {isSending ? "Sending..." : "Send"}
          </button>
        </form>

        {error ? <p className="error-banner">{error}</p> : null}
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

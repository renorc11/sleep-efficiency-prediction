"use client";

import { FormEvent, useState } from "react";
import { chat } from "../lib/api";

type Message = {
  role: "user" | "assistant";
  content: string;
};

export default function SleepAssistant() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Hi! I'm your sleep wellness assistant. Ask me about sleep habits, sleep efficiency, or ways to improve your sleep routine.",
    },
  ]);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const message = input.trim();

    if (!message || loading) {
      return;
    }

    setMessages((previous) => [
      ...previous,
      { role: "user", content: message },
    ]);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const response = await chat(message);

      setMessages((previous) => [
        ...previous,
        { role: "assistant", content: response.reply },
      ]);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to contact the sleep assistant.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <section
      aria-labelledby="sleep-assistant-title"
      style={{
        maxWidth: "850px",
        margin: "40px auto",
        padding: "24px",
        border: "1px solid #e5e7eb",
        borderRadius: "16px",
        background: "#ffffff",
        color: "#111827",
      }}
    >
      <div style={{ marginBottom: "20px" }}>
        <p
          style={{
            margin: "0 0 8px",
            fontSize: "12px",
            fontWeight: 700,
            letterSpacing: "1px",
            textTransform: "uppercase",
            color: "#64748b",
          }}
        >
          Powered by BioMistral
        </p>

        <h2
          id="sleep-assistant-title"
          style={{ margin: "0 0 8px", fontSize: "24px" }}
        >
          Sleep Wellness Assistant
        </h2>

        <p style={{ margin: 0, color: "#64748b", fontSize: "14px" }}>
          Get practical guidance about sleep habits and sleep efficiency.
        </p>
      </div>

      <div
        aria-live="polite"
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "12px",
          minHeight: "180px",
          maxHeight: "400px",
          overflowY: "auto",
          padding: "16px",
          borderRadius: "12px",
          background: "#f8fafc",
        }}
      >
        {messages.map((message, index) => (
          <div
            key={`${index}-${message.role}`}
            style={{
              alignSelf:
                message.role === "user" ? "flex-end" : "flex-start",
              maxWidth: "90%",
              padding: "12px 16px",
              borderRadius: "12px",
              background:
                message.role === "user" ? "#dbeafe" : "#ffffff",
              border: "1px solid #e5e7eb",
              whiteSpace: "pre-wrap",
              overflowWrap: "anywhere",
              lineHeight: 1.6,
              fontSize: "14px",
            }}
          >
            <strong
              style={{
                display: "block",
                marginBottom: "4px",
                fontSize: "12px",
                color: "#64748b",
              }}
            >
              {message.role === "user" ? "You" : "Sleep Assistant"}
            </strong>

            {message.content}
          </div>
        ))}

        {loading && (
          <p style={{ margin: 0, color: "#64748b", fontSize: "14px" }}>
            Sleep Assistant is thinking…
          </p>
        )}
      </div>

      {error && (
        <p role="alert" style={{ color: "#b91c1c", fontSize: "14px" }}>
          {error}
        </p>
      )}

      <form
        onSubmit={handleSubmit}
        style={{
          display: "flex",
          gap: "10px",
          marginTop: "16px",
        }}
      >
        <input
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Ask a question about your sleep…"
          aria-label="Your sleep question"
          maxLength={4000}
          disabled={loading}
          style={{
            flex: 1,
            minWidth: 0,
            padding: "12px 14px",
            border: "1px solid #cbd5e1",
            borderRadius: "10px",
            fontSize: "14px",
          }}
        />

        <button
          type="submit"
          disabled={loading || !input.trim()}
          style={{
            padding: "12px 18px",
            border: 0,
            borderRadius: "10px",
            background: loading || !input.trim() ? "#94a3b8" : "#1d4ed8",
            color: "#ffffff",
            fontWeight: 600,
            cursor: loading || !input.trim() ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "Sending…" : "Send"}
        </button>
      </form>

      <p
        style={{
          margin: "12px 0 0",
          color: "#64748b",
          fontSize: "12px",
          lineHeight: 1.5,
        }}
      >
        This assistant provides general sleep-wellness education, not a
        diagnosis or a substitute for professional medical advice.
      </p>
    </section>
  );
}

import { useCallback, useEffect, useRef, useState } from 'react';
import { buildChatContext } from '../lib/chatContext.js';
import { openChatStream } from '../lib/chatStream.js';

export default function useAppState() {
  const [health, setHealth] = useState(null);
  const [backendError, setBackendError] = useState(false);
  const [checkingHealth, setCheckingHealth] = useState(true);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [context, setContext] = useState('');
  const [contextOpen, setContextOpen] = useState(false);
  const healthRequest = useRef(null);
  const generation = useRef(null);
  const nextId = useRef(0);

  const fetchHealth = useCallback(async () => {
    if (healthRequest.current) return;
    const controller = new AbortController();
    healthRequest.current = controller;
    setCheckingHealth(true);

    try {
      const response = await fetch('/api/health', { signal: controller.signal, cache: 'no-store' });
      if (!response.ok) throw new Error(`Health check failed (${response.status}).`);
      const result = await response.json();
      if (healthRequest.current !== controller) return;
      setHealth(result);
      setBackendError(false);
    } catch {
      if (healthRequest.current === controller) setBackendError(true);
    } finally {
      if (healthRequest.current === controller) {
        healthRequest.current = null;
        setCheckingHealth(false);
      }
    }
  }, []);

  const detachGeneration = useCallback(() => {
    const request = generation.current;
    generation.current = null;
    if (request) {
      cancelAnimationFrame(request.frame);
      request.controller.abort();
    }
  }, []);

  useEffect(() => {
    fetchHealth();
    return () => {
      const request = healthRequest.current;
      healthRequest.current = null;
      request?.abort();
      detachGeneration();
    };
  }, [fetchHealth, detachGeneration]);

  useEffect(() => {
    const poll = setInterval(fetchHealth, !backendError && health?.status === 'ready' ? 10000 : 2500);
    return () => clearInterval(poll);
  }, [health?.status, backendError, fetchHealth]);

  const updateAssistant = (request, details) => {
    const content = request.content;
    setMessages((current) => current.map((message) =>
      message.id === request.id ? { ...message, content, ...details } : message,
    ));
  };

  function finish(request, details = {}) {
    if (generation.current !== request) return;
    generation.current = null;
    cancelAnimationFrame(request.frame);
    updateAssistant(request, { isStreaming: false, ...details });
  }

  function stopGenerating() {
    const request = generation.current;
    if (!request) return;
    finish(request, { stoppedReason: 'cancelled' });
    request.controller.abort();
  }

  function handleNewChat() {
    detachGeneration();
    setMessages([]);
    setInput('');
    setContext('');
    setContextOpen(false);
  }

  const isReady = health?.status === 'ready' && !backendError;

  async function sendMessage(prompt = input, reference = context) {
    const message = prompt.trim();
    const requestContext = reference.trim();
    if (!message || !isReady || generation.current) return;

    const request = { id: ++nextId.current, controller: new AbortController(), content: '', frame: 0 };
    generation.current = request;
    setInput('');
    setMessages((current) => [
      ...current,
      { id: `prompt-${request.id}`, role: 'user', content: message, context: requestContext },
      { id: request.id, role: 'assistant', content: '', isStreaming: true },
    ]);

    try {
      const modelContext = buildChatContext(messages, message, requestContext, health?.max_new_tokens);
      const events = await openChatStream(message, modelContext, request.controller.signal);
      for await (const event of events) {
        if (generation.current !== request) return;
        if (event.type === 'token') {
          request.content += event.text ?? '';
          if (!request.frame) {
            request.frame = requestAnimationFrame(() => {
              request.frame = 0;
              if (generation.current === request) updateAssistant(request);
            });
          }
        } else if (event.type === 'done') {
          finish(request, {
            tokens: event.tokens,
            tokensPerSecond: event.tokens_per_second,
            stoppedReason: event.stopped_reason,
          });
          return;
        } else if (event.type === 'error') {
          finish(request, { error: event.message || 'Generation failed.' });
          return;
        }
      }
      finish(request, { error: 'The connection closed before the response finished.' });
    } catch (error) {
      finish(request, { error: error.message || 'Unable to reach the backend.' });
    }
  }

  return {
    health, backendError, checkingHealth, fetchHealth, isReady,
    messages, input, setInput, context, setContext, contextOpen, setContextOpen,
    stopGenerating, handleNewChat, sendMessage,
  };
}

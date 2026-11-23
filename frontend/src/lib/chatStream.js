export async function* readChatEvents(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let data = [];
  let finished = false;

  try {
    while (!finished) {
      const chunk = await reader.read();
      finished = chunk.done;
      buffer += decoder.decode(chunk.value, { stream: !finished });

      // Keep a trailing CR until the next read can distinguish it from CRLF.
      const lines = buffer.split(finished ? /\r\n|\r|\n/ : /\r\n|\r(?!$)|\n/);
      buffer = lines.pop();
      for (const line of lines) {
        if (line === '') {
          if (!data.length) continue;
          const payload = data.join('\n');
          data = [];
          yield JSON.parse(payload);
          continue;
        }

        if (line === 'data') data.push('');
        if (line.startsWith('data:')) {
          data.push(line.slice(5).replace(/^ /, ''));
        }
      }
    }
  } finally {
    if (!finished) await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

export async function openChatStream(message, context, signal) {
  const response = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ message, context }),
    signal,
  });

  if (!response.ok) {
    const { detail } = await response.json().catch(() => ({}));
    const message = detail?.message ?? detail?.[0]?.msg;
    throw new Error(message ?? `Chat request failed (${response.status}).`);
  }

  if (!response.body || !response.headers.get('content-type')?.includes('text/event-stream')) {
    throw new Error('The backend did not return an event stream.');
  }

  return readChatEvents(response.body);
}

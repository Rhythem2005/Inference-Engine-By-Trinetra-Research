const encoder = new TextEncoder();
const byteLength = (text) => encoder.encode(text).length;
const historyHeading = 'Conversation so far:\n';

export function buildChatContext(messages, message, reference, maxNewTokens = 256) {
  if (!messages.length) return reference;

  const referenceSection = reference ? `\n\nReference context:\n${reference}` : '';
  const contextEnd = `${referenceSection}\n\nLatest user message:\n${message}`;

  const promptWithoutTurns = `### Instruction:\n${message}\n\n### Input:\n${historyHeading}${contextEnd}\n\n### Response:\n`;
  let available = 2048 - maxNewTokens - byteLength(promptWithoutTurns);
  if (available <= 0) return reference;

  const turns = [];
  let reply = '';
  for (let index = messages.length - 1; index >= 0; index--) {
    const { role, content } = messages[index];
    if (role === 'assistant') {
      reply = content;
      continue;
    }

    const user = `User: ${content}`;
    const turn = reply ? `${user}\nAssistant: ${reply}` : user;
    const cost = byteLength(turn) + (turns.length ? 2 : 0);
    if (cost > available) {
      if (!turns.length && byteLength(user) <= available) turns.push(user);
      break;
    }
    turns.push(turn);
    available -= cost;
    reply = '';
  }

  return turns.length ? `${historyHeading}${turns.reverse().join('\n\n')}${contextEnd}` : reference;
}

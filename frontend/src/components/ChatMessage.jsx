import React, { memo, useId, useState } from 'react';
import { motion } from 'framer-motion';
import { ChevronDown, Paperclip } from 'lucide-react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import AgentAvatar from './AgentAvatar.jsx';
import CodeBlock from './CodeBlock.jsx';
import CopyButton from './CopyButton.jsx';

const enter = { initial: { opacity: 0, y: 6 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.2 } };
const markdownPlugins = [remarkGfm];
const markdownComponents = {
  pre: CodeBlock,
  img: ({ alt }) => alt || null,
  table: ({ node, ...props }) => <div className="table-scroll"><table {...props} /></div>,
};

export default memo(function ChatMessage({ message, mascot }) {
  const [showContext, setShowContext] = useState(false);
  const contextId = useId();

  if (message.role === 'user') {
    return (
      <motion.article className="user-message" aria-label="You" {...enter}>
        <div className="user-content">{message.content}</div>
        {message.context && (
          <>
            <button className="text-button message-context-toggle" type="button" aria-expanded={showContext}
              aria-controls={contextId} onClick={() => setShowContext(!showContext)}>
              <Paperclip size={13} aria-hidden="true" /> Reference context <ChevronDown size={12} aria-hidden="true" />
            </button>
            <div id={contextId} className="message-reference" hidden={!showContext}>{message.context}</div>
          </>
        )}
      </motion.article>
    );
  }

  const pending = message.isStreaming && !message.content;
  const writing = Boolean(message.isStreaming && message.content);
  const stopped = message.stoppedReason === 'cancelled';
  const hasTelemetry = typeof message.tokens === 'number' || typeof message.tokensPerSecond === 'number';
  let agentState = 'done';
  if (message.error) agentState = 'alert';
  else if (pending) agentState = 'thinking';
  else if (writing) agentState = 'writing';

  return (
    <motion.article className="assistant-message" aria-label="KAAL" {...enter}>
      <span className="assistant-avatar"><AgentAvatar state={agentState} variant={mascot} /></span>
      <div className="assistant-content">
        <div className="assistant-name">KAAL</div>
        {pending && (
          <div className="thinking">
            <span className="thinking-dots" aria-hidden="true"><i /><i /><i /></span>
            Thinking…
          </div>
        )}
        {message.content && (
          <div className="markdown" data-writing={writing}>
            <Markdown components={markdownComponents} remarkPlugins={markdownPlugins} skipHtml>{message.content}</Markdown>
          </div>
        )}
        {message.error && <p className="message-error" role="alert">{message.error}</p>}
        {!message.isStreaming && !message.error && !message.content && (
          <p className="empty-response">{stopped ? 'Generation stopped.' : 'No response was generated.'}</p>
        )}
        {!message.isStreaming && (hasTelemetry || message.content) && (
          <motion.div className="message-footer" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.25 }}>
            {message.content && <CopyButton content={message.content} target="response" />}
            <span className="telemetry">
              {typeof message.tokensPerSecond === 'number' && <span>{message.tokensPerSecond.toFixed(1)} tokens/sec</span>}
              {typeof message.tokens === 'number' && <span>{message.tokens} {message.tokens === 1 ? 'token' : 'tokens'}</span>}
              {stopped && <span>Stopped</span>}
            </span>
          </motion.div>
        )}
      </div>
    </motion.article>
  );
});

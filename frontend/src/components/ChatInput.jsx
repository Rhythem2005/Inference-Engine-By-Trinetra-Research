import React, { useLayoutEffect, useRef } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { ArrowUp, Check, ChevronDown, CornerDownLeft, Paperclip, Square } from 'lucide-react';

import ModelPicker from './ModelPicker.jsx';

export default function ChatInput({
  input, setInput, context, setContext, contextOpen, setContextOpen,
  onSend, onStop, isGenerating, disabled, offline,
}) {
  const prompt = useRef(null);
  const hasContext = Boolean(context.trim());

  useLayoutEffect(() => {
    const field = prompt.current;
    field.style.height = 'auto';
    field.style.height = `${field.scrollHeight}px`;
  }, [input]);

  function submit(event) {
    event.preventDefault();
    if (!disabled && !isGenerating && input.trim()) onSend();
  }

  return (
    <div className="dock">
      <form className="composer" onSubmit={submit} aria-label="Chat composer">
        <AnimatePresence initial={false}>
          {contextOpen && (
            <motion.div className="context-reveal" id="reference-panel"
              initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.18 }}>
              <div className="context-editor">
                <div className="context-heading">
                  <label htmlFor="reference-context">Reference context</label>
                  {hasContext && <button className="text-button" type="button" onClick={() => setContext('')}>Clear</button>}
                </div>
                <textarea id="reference-context" rows={3} maxLength={32000} value={context}
                  onChange={(event) => setContext(event.target.value)}
                  placeholder="Paste a passage, notes, or code to include with your prompt." />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        <textarea ref={prompt} className="prompt-input" aria-label="Message" rows={1} maxLength={16000}
          value={input} onChange={(event) => setInput(event.target.value)}
          placeholder={offline ? 'Waiting for the local backend…' : 'Message KAAL…'}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) submit(event);
          }} />
        <div className="composer-tools">
          <button className="context-toggle" type="button" aria-expanded={contextOpen}
            aria-controls={contextOpen ? 'reference-panel' : undefined} onClick={() => setContextOpen(!contextOpen)}>
            <Paperclip size={15} aria-hidden="true" /><span>Add Reference Context</span>
            {hasContext && <Check className="context-check" size={13} aria-label="Context attached" />}
            <ChevronDown className="context-chevron" size={12} aria-hidden="true" />
          </button>
          <div className="send-tools">
            <ModelPicker />
            {!isGenerating && <span className="key-hint"><CornerDownLeft size={12} aria-hidden="true" /> Enter to send</span>}
            {isGenerating ? (
              <motion.button className="stop-button" type="button" whileTap={{ scale: 0.95 }} onClick={onStop}>
                <Square size={11} fill="currentColor" aria-hidden="true" /> Stop
              </motion.button>
            ) : (
              <motion.button className="send-button" type="submit" whileTap={{ scale: 0.95 }}
                disabled={disabled || !input.trim()} aria-label="Send message">
                <ArrowUp size={20} strokeWidth={2.1} aria-hidden="true" />
              </motion.button>
            )}
          </div>
        </div>
      </form>
    </div>
  );
}

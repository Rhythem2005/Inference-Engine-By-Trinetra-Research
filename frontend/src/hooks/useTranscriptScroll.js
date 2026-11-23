import { useLayoutEffect, useRef, useState } from 'react';

export default function useTranscriptScroll(messages, visible) {
  const viewport = useRef(null);
  const content = useRef(null);
  const following = useRef(true);
  const lastTop = useRef(0);
  const [showLatest, setShowLatest] = useState(false);
  const hasMessages = visible && messages.length > 0;

  function alignToLatest() {
    const element = viewport.current;
    if (!element) return;
    element.scrollTop = element.scrollHeight;
    lastTop.current = element.scrollTop;
  }

  function follow(value) {
    following.current = value;
    setShowLatest(!value);
  }

  function onScroll(event) {
    const element = event.currentTarget;
    const gap = element.scrollHeight - element.clientHeight - element.scrollTop;
    if (element.scrollTop < lastTop.current - 1) follow(false);
    else if (gap <= 3) follow(true);
    lastTop.current = element.scrollTop;
  }

  useLayoutEffect(() => {
    if (!hasMessages) {
      follow(true);
      return;
    }

    const observer = new ResizeObserver(() => {
      if (following.current) alignToLatest();
    });
    observer.observe(viewport.current);
    observer.observe(content.current);
    return () => observer.disconnect();
  }, [hasMessages]);

  useLayoutEffect(() => {
    if (following.current) alignToLatest();
  }, [messages]);

  return {
    viewport, content, showLatest, onScroll,
    onWheel: (event) => {
      if (event.deltaY < 0 && event.currentTarget.scrollTop > 0) follow(false);
    },
    jumpToLatest: () => {
      follow(true);
      alignToLatest();
    },
  };
}

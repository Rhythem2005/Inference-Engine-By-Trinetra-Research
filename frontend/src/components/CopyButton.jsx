import React, { useEffect, useState } from 'react';
import { Check, Copy } from 'lucide-react';

export default function CopyButton({ content, target }) {
  const [result, setResult] = useState('');
  useEffect(() => {
    if (!result) return;
    const timeout = setTimeout(() => setResult(''), 1800);
    return () => clearTimeout(timeout);
  }, [result]);

  async function copy() {
    try {
      await navigator.clipboard.writeText(content);
      setResult('Copied');
    } catch {
      setResult('Copy failed');
    }
  }

  return (
    <button className="text-button copy-button" type="button" onClick={copy} aria-label={result || `Copy ${target}`}>
      {result === 'Copied' ? <Check size={14} aria-hidden="true" /> : <Copy size={14} aria-hidden="true" />}
      <span aria-live="polite">{result || 'Copy'}</span>
    </button>
  );
}

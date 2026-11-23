import React, { isValidElement, useEffect, useMemo, useState } from 'react';
import CopyButton from './CopyButton.jsx';

const labels = {
  c: 'C', cpp: 'C++', css: 'CSS', go: 'Go', html: 'HTML', java: 'Java', javascript: 'JavaScript',
  js: 'JavaScript', json: 'JSON', jsx: 'JSX', md: 'Markdown', py: 'Python', python: 'Python',
  rust: 'Rust', sh: 'Shell', sql: 'SQL', ts: 'TypeScript', tsx: 'TSX', typescript: 'TypeScript',
  yml: 'YAML', yaml: 'YAML',
};

export default function CodeBlock({ children }) {
  const code = isValidElement(children) ? children.props : {};
  const source = typeof code.children === 'string' ? code.children.replace(/\n$/, '') : '';
  const language = code.className?.match(/\blanguage-([\w+-]+)/)?.[1].toLowerCase();
  const label = Object.hasOwn(labels, language) ? labels[language] : language;
  const [highlighter, setHighlighter] = useState(null);

  useEffect(() => {
    if (!language || highlighter) return;
    let cancelled = false;
    import('../lib/highlight.js').then(({ highlightCode }) => {
      if (!cancelled) setHighlighter(() => highlightCode);
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [language, highlighter]);

  const html = useMemo(() => language ? highlighter?.(source, language) : null, [highlighter, language, source]);

  return (
    <div className="code-block">
      <div className="code-header">
        <span>{label ?? 'Code'}</span>
        <CopyButton content={source} target="code" />
      </div>
      <pre tabIndex={0} aria-label={`${label ?? 'Plain text'} code block`}>
        {html == null ? <code>{source}</code> : <code dangerouslySetInnerHTML={{ __html: html }} />}
      </pre>
    </div>
  );
}

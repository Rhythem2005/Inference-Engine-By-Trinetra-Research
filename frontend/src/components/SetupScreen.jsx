import React from 'react';
import { motion } from 'framer-motion';
import { File, RotateCw } from 'lucide-react';
import AgentAvatar from './AgentAvatar.jsx';

const files = ['best_model.pt', 'tokenizer.json'];

export default function SetupScreen({ missingFiles = [], onRetry, checking, mascot }) {
  return (
    <main className="setup-shell">
      <motion.section className="setup-card" aria-labelledby="setup-title"
        initial={{ opacity: 0, y: 7 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.22 }}>
        <AgentAvatar className="setup-mascot" state="alert" variant={mascot || undefined} />
        <h1 id="setup-title">Model weights missing</h1>
        <p>Place these files in <code>./weights/</code>, then restart the KAAL backend.</p>
        <ul className="setup-files">
          {files.map((file) => {
            const missing = missingFiles.length === 0 || missingFiles.some((path) => path.split('/').pop() === file);
            return (
              <li key={file}><File size={15} aria-hidden="true" /><code>{file}</code>
                <span data-missing={missing}>{missing ? 'Missing' : 'Found'}</span>
              </li>
            );
          })}
        </ul>
        <button className="retry-button" type="button" onClick={onRetry} disabled={checking}>
          <RotateCw size={14} aria-hidden="true" /> {checking ? 'Checking…' : 'Check status'}
        </button>
      </motion.section>
    </main>
  );
}

import React from 'react';
import { Plus } from 'lucide-react';
import trinetraLogo from '../trinetra.svg';

export default function Navbar({ health, backendError, onNewChat, onAbout }) {
  const ready = !backendError && health?.status === 'ready';
  const device = health?.device === 'mps' ? 'Apple Silicon MPS' : (health?.device || 'CPU').toUpperCase();
  let status = 'Loading…';
  if (backendError) status = 'Backend offline';
  else if (health?.status === 'weights_missing') status = 'Weights Missing';
  else if (ready) status = `Ready • ${device}`;

  return (
    <header className="masthead">
      <div className="masthead-inner">
        <div className="wordmark">
          <img className="brand-mark" src={trinetraLogo} alt="" />
          <span>Trinetra Research</span>
        </div>
        <div className="masthead-actions">
          <span className="health-badge" data-ready={ready} role="status">
            <span className="status-dot" aria-hidden="true" /><span>{status}</span>
          </span>
          <button className="about-button" type="button" onClick={() => onAbout('philosophy')}>Philosophy</button>
          <button className="about-button" type="button" onClick={() => onAbout('founder')}>Founder</button>
          <button className="new-chat" type="button" onClick={onNewChat}>
            <Plus size={16} strokeWidth={1.8} aria-hidden="true" /> New Chat
          </button>
        </div>
      </div>
    </header>
  );
}

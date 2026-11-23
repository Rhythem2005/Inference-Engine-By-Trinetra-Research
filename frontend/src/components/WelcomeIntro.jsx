import React from 'react';
import { ArrowRight } from 'lucide-react';
import AgentAvatar from './AgentAvatar.jsx';

export default function WelcomeIntro({ onContinue }) {
  return (
    <main className="welcome intro" aria-labelledby="intro-title">
      <div className="intro-panel">
        <div className="intro-copy">
          <h1 id="intro-title">Welcome to<br />Trinetra Research.</h1>
          <p>A quiet place to ask questions, play with ideas, and explore a language model running on your device.</p>
          <button className="intro-action" type="button" onClick={onContinue}>
            Get started <ArrowRight size={17} aria-hidden="true" />
          </button>
        </div>
        <div className="intro-portrait" aria-hidden="true">
          <span className="portrait-spark spark-top" />
          <AgentAvatar />
          <span className="portrait-spark spark-bottom" />
        </div>
      </div>
    </main>
  );
}

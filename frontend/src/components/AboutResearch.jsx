import React, { useEffect, useRef } from 'react';
import { X } from 'lucide-react';

export default function AboutResearch({ section, onClose }) {
  const dialogRef = useRef(null);
  const title = section === 'philosophy' ? 'Philosophy' : 'Founder';

  useEffect(() => {
    if (!dialogRef.current.open) dialogRef.current.showModal();
  }, []);

  return (
    <dialog className="research-dialog" ref={dialogRef} onClose={onClose} aria-labelledby="research-title">
      <header className="research-bar">
        <strong id="research-title">TRINETRA AI RESEARCH</strong>
        <span className="research-panel-name">{title}</span>
        <form method="dialog"><button className="notice-close" aria-label={`Close ${title}`}><X size={18} aria-hidden="true" /></button></form>
      </header>
      <div className="research-content">
        {section === 'philosophy' && <section id="research-philosophy" aria-labelledby="philosophy-title">
          <p className="research-label">Philosophy</p>
          <h2 id="philosophy-title">Our identity: Trinetra / <span lang="hi">त्रिनेत्र</span></h2>
          <p className="research-lead">Trinetra means “three-eyed.” The name draws on Mahadev’s third eye as a symbol of curiosity, deeper understanding, and looking beyond the obvious.</p>
          <h3>The meaning of Trinetra</h3>
          <p>Trinetra (<span lang="hi">त्रिनेत्र</span>) means three-eyed and is associated with Shiva, or Mahadev. In this project, the third eye is a personal metaphor for looking more closely: asking how a system works, examining assumptions, and seeking understanding beyond its surface.</p>
          <h3>Curiosity, with cultural roots</h3>
          <p>Shaivite traditions and Indian philosophical inquiry inform the project’s name and visual world. <span lang="hi">जिज्ञासा</span> — curiosity — describes its approach to learning. Technical explanations and model claims are grounded in implementation and evaluation.</p>
        </section>}
        {section === 'founder' && <section id="research-founder" aria-labelledby="founder-title">
          <h2 id="founder-title">About the founder</h2>
          <p>Trinetra AI Research started as an independent project to learn how language models are built, trained, and evaluated. The Website link in the header has more about the people and projects behind the research.</p>
        </section>}
      </div>
    </dialog>
  );
}

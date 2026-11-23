import React, { useEffect, useRef } from 'react';
import { X } from 'lucide-react';

const work = [
  ['Language models', 'Developed 30M and 304M-parameter GPT-style language models, with hands-on experience in Transformer architectures, pretraining and evaluation.'],
  ['Computer vision', 'An adaptive object-detection pipeline exploring efficiency and accuracy trade-offs.'],
  ['Neuroscience-inspired computing', 'Work investigating visual escape behavior in Drosophila.'],
  ['Applied AI', 'FarmWise, an agricultural AI platform incorporating disease detection and yield prediction.'],
  ['Full-stack engineering', 'FlavourFind, a real-time food-delivery comparison platform using React, FastAPI and Playwright.'],
  ['AI systems', 'Experimentation with local language models, inference systems and AI-agent architectures.'],
];

export default function AboutResearch({ section, onClose }) {
  const dialogRef = useRef(null);
  const title = section === 'philosophy' ? 'Philosophy' : 'Founder';

  useEffect(() => {
    if (!dialogRef.current.open) dialogRef.current.showModal();
  }, []);

  return (
    <dialog className="research-dialog" ref={dialogRef} onClose={onClose} aria-labelledby="research-title">
      <header className="research-bar">
        <strong id="research-title">TRINETRA RESEARCH</strong>
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
          <h3>The person behind the project</h3>
          <p>Trinetra Research is an independent Indian AI initiative created by a fourth-year student for education, experimentation, and hands-on understanding of foundation models. Its purpose is to understand the systems through implementation, not to compete with frontier AI labs or existing open-source models.</p>
          <h3>Research within available resources</h3>
          <p>Most of the work relies on accessible and free GPUs. Pretraining, evaluation, instruction fine-tuning and preparing a release can each take considerable time. I’d rather document what I understand and what remains uncertain than work toward a release date I cannot support.</p>
          <h3>Learning toward a research career</h3>
          <p>I want to turn this work into the skills needed for AI research and engineering: implementing a model, evaluating it carefully and explaining what the results do and do not show.</p>
        </section>}
        {section === 'founder' && <section id="research-founder" aria-labelledby="founder-title">
          <p className="research-label">The person behind the work</p>
          <p className="founder-hindi" lang="hi">जिज्ञासा से समझ तक।</p>
          <h2 id="founder-title">Rhythem <em>Sabharwal.</em></h2>
          <p>Founder, TRINETRA RESEARCH<br />New Delhi, India</p>
          <p className="research-lead">Understanding begins with building.</p>
          <p>Trinetra Research is an independent, student-led initiative for studying language models through implementation and experimentation. Rhythem Sabharwal works across language-model training, computer vision and full-stack systems. Trinetra is where he documents the model-building side of that work.</p>
          <h3 className="research-subtitle">Research and <em>engineering.</em></h3>
          <p>Independent research and engineering experiences that inform the work at Trinetra.</p>
          <dl className="research-grid">
            {work.map(([title, description]) => <div key={title}><dt>{title}</dt><dd>{description}</dd></div>)}
          </dl>
          <p className="research-label">Education</p>
          <h3 className="research-subtitle">Learning, <em>in parallel.</em></h3>
          <dl className="research-grid">
            <div><dt>BTech Computer Science and Engineering</dt><dd>Bhagwan Parshuram Institute of Technology<br /><time>2023–2027</time></dd></div>
            <div><dt>Major in Artificial Intelligence</dt><dd>Indian Institute Of Technology Ropar<br /><time>2024–2026</time></dd></div>
          </dl>
          <h3 className="research-subtitle">Get in touch.</h3>
          <p>For research questions, or a closer look at Rhythem’s other projects.</p>
        </section>}
      </div>
    </dialog>
  );
}

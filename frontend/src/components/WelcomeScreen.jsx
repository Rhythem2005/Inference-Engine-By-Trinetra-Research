import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowUpRight, Film, Leaf, MapPin, Trophy } from 'lucide-react';
import AgentAvatar from './AgentAvatar.jsx';
import { mascots } from '../lib/mascots.js';

const prompts = [
  { icon: Trophy, label: "Suggest me some sports?", title: "Find a sport to try", message: "Suggest me some sports?" },
  { icon: Leaf, label: "Explain photosynthesis", title: "How plants use sunlight", message: "Explain photosynthesis in simple terms." },
  { icon: MapPin, label: "Find a story setting", title: "An abandoned cinema", message: "Give one unusual location for a story about an abandoned cinema." },
  { icon: Film, label: "Suggest late-night movies", title: "Find something to watch", message: "Suggest three movies for a late-night watch and briefly explain why each fits." },
];

export default function WelcomeScreen({ name, mascot, onSaveProfile, onSelectPrompt, disabled }) {
  const [draftName, setDraftName] = useState(name);
  const [draftMascot, setDraftMascot] = useState(mascot || mascots[0].id);
  const [editing, setEditing] = useState(false);
  const showProfile = !name || !mascot || editing;

  function submitProfile(event) {
    event.preventDefault();
    const value = draftName.trim();
    if (!value) return;
    onSaveProfile(value, draftMascot);
    setEditing(false);
  }
  return (
    <main className="welcome">
      <motion.div className="welcome-inner" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>
        <section className="welcome-greeting" aria-labelledby="welcome-title">
          <div className="mascot-portrait">
            <AgentAvatar variant={showProfile ? draftMascot : mascot} />
            <span className="portrait-spark spark-top" aria-hidden="true" />
            <span className="portrait-spark spark-bottom" aria-hidden="true" />
          </div>
          <h1 id="welcome-title">{name ? `Hi, ${name}.` : 'What should I call you?'}</h1>
          {showProfile ? (
            <form className="profile-form" onSubmit={submitProfile}>
              {!name && <input aria-label="Your name" autoComplete="nickname" maxLength={48} placeholder="Your name or nickname"
                value={draftName} onChange={(event) => setDraftName(event.target.value)} />}
              <fieldset className="mascot-options">
                <legend>Choose your mascot</legend>
                <div>
                  {mascots.map(({ id, name: mascotName }) => (
                    <label key={id}>
                      <input className="sr-only" type="radio" name="mascot" value={id} checked={draftMascot === id}
                        onChange={() => setDraftMascot(id)} />
                      <span className="mascot-option"><AgentAvatar variant={id} state="done" />{mascotName}</span>
                    </label>
                  ))}
                </div>
              </fieldset>
              <button type="submit" disabled={!draftName.trim()}>{editing ? 'Save mascot' : 'Continue'}</button>
            </form>
          ) : (
            <button type="button" className="change-mascot" onClick={() => setEditing(true)}>Change mascot</button>
          )}
        </section>

        {!showProfile && <section className="starters" aria-labelledby="starters-heading">
          <h2 id="starters-heading">Try a prompt</h2>
          <div className="prompt-grid">
            {prompts.map(({ icon: Icon, label, title, message }) => (
              <button key={label} className="prompt-card" type="button" disabled={disabled}
                onClick={() => onSelectPrompt(message, '')}>
                <span className="prompt-icon"><Icon size={16} strokeWidth={1.75} aria-hidden="true" /></span>
                <span className="prompt-description"><strong>{label}</strong><span>{title}</span></span>
                <ArrowUpRight className="prompt-arrow" size={15} strokeWidth={1.6} aria-hidden="true" />
              </button>
            ))}
          </div>
        </section>}
      </motion.div>
    </main>
  );
}

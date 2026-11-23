import React, { lazy, Suspense, useState } from 'react';
import { MotionConfig, motion } from 'framer-motion';
import { ArrowDown, LockKeyhole } from 'lucide-react';
import useAppState from './hooks/useAppState.js';
import useTranscriptScroll from './hooks/useTranscriptScroll.js';
import Navbar from './components/Navbar.jsx';
import WelcomeScreen from './components/WelcomeScreen.jsx';
import SetupScreen from './components/SetupScreen.jsx';
import ChatMessage from './components/ChatMessage.jsx';
import ChatInput from './components/ChatInput.jsx';
import { mascots } from './lib/mascots.js';

const ModelGuide = lazy(() => import('./components/ModelGuide.jsx'));
const AboutResearch = lazy(() => import('./components/AboutResearch.jsx'));
const WelcomeIntro = lazy(() => import('./components/WelcomeIntro.jsx'));

const nameKey = 'kaal.displayName';
const mascotKey = 'trinetra.mascot';

function readPreference(key) {
  try {
    return localStorage.getItem(key) || '';
  } catch {
    return '';
  }
}

export default function App() {
  const [aboutSection, setAboutSection] = useState(null);
  const [introContinued, setIntroContinued] = useState(false);
  const [name, setName] = useState(() => readPreference(nameKey).trim().slice(0, 48));
  const [mascot, setMascot] = useState(() => {
    const saved = readPreference(mascotKey);
    return mascots.some(({ id }) => id === saved) ? saved : '';
  });

  function saveProfile(displayName, mascotId) {
    setName(displayName);
    setMascot(mascotId);
    try {
      localStorage.setItem(nameKey, displayName);
      localStorage.setItem(mascotKey, mascotId);
    } catch {
      // Preferences still work for this visit if storage is unavailable.
    }
  }
  const chat = useAppState();
  const active = chat.messages.at(-1);
  const isGenerating = Boolean(active?.isStreaming);
  const showSetup = !isGenerating && !chat.backendError && chat.health?.status === 'weights_missing';
  const scroll = useTranscriptScroll(chat.messages, !showSetup);
  const showIntro = !name && !introContinued;

  return (
    <MotionConfig reducedMotion="user">
      <div className="chat-app">
        <Navbar health={chat.health} backendError={chat.backendError} onNewChat={chat.handleNewChat} onAbout={setAboutSection} />
        {aboutSection && (
          <Suspense fallback={null}>
            <AboutResearch section={aboutSection} onClose={() => setAboutSection(null)} />
          </Suspense>
        )}

        <Suspense fallback={null}>
          <ModelGuide compact={chat.messages.length > 0} />
        </Suspense>

        {showIntro ? (
          <Suspense fallback={null}>
            <WelcomeIntro onContinue={() => setIntroContinued(true)} />
          </Suspense>
        ) : showSetup ? (
          <SetupScreen missingFiles={chat.health.missing_files} checking={chat.checkingHealth} onRetry={chat.fetchHealth} mascot={mascot} />
        ) : (
          <>
            {chat.messages.length ? (
              <main className="transcript-shell">
                <div className="sr-only" role="status">
                  {isGenerating ? (active.content ? 'Writing a reply' : 'Thinking') : ''}
                </div>
                <div className="transcript" ref={scroll.viewport} onScroll={scroll.onScroll}
                  onWheel={scroll.onWheel} tabIndex={0} aria-label="Conversation">
                  <div className="message-list" ref={scroll.content}>
                    {chat.messages.map((message) => <ChatMessage key={message.id} message={message} mascot={mascot} />)}
                  </div>
                </div>
                {scroll.showLatest && (
                  <motion.button className="latest-button" type="button" onClick={scroll.jumpToLatest}
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <ArrowDown size={14} aria-hidden="true" /> Latest response
                  </motion.button>
                )}
              </main>
            ) : (
              <WelcomeScreen name={name} mascot={mascot} onSaveProfile={saveProfile} onSelectPrompt={chat.sendMessage} disabled={!chat.isReady} />
            )}

            {name && mascot && <ChatInput
              input={chat.input} setInput={chat.setInput}
              context={chat.context} setContext={chat.setContext}
              contextOpen={chat.contextOpen} setContextOpen={chat.setContextOpen}
              onSend={() => {
                scroll.jumpToLatest();
                chat.sendMessage();
              }}
              onStop={chat.stopGenerating}
              isGenerating={isGenerating} disabled={!chat.isReady}
              offline={chat.backendError}
            />}
          </>
        )}
        <footer className="app-footer">
          <span>AI can make mistakes. Double-check important information.</span>
          <span className="local-note"><LockKeyhole size={12} aria-hidden="true" /> Runs locally</span>
        </footer>
      </div>
    </MotionConfig>
  );
}

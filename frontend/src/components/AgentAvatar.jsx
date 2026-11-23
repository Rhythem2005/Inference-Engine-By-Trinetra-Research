import React from 'react';

export default function AgentAvatar({ state = 'idle', className = '', variant = 'robot' }) {
  const alert = state === 'alert';
  return (
    <svg className={`agent ${className}`} data-agent-state={state} data-mascot={variant} viewBox="0 0 96 96" fill="none" aria-hidden="true">
      <ellipse className="agent-shadow" cx="46" cy="83" rx="25" ry="4" fill="#6A3B24" opacity=".13" />
      <g className="agent-body">
        <path d="M35 72q-2 8-8 7M56 72q2 8 8 6" stroke="#50382E" strokeWidth="2.8" strokeLinecap="round" />
        <path d="M35 66q-9-1-13-7m42 6 7-6" stroke="#50382E" strokeWidth="2.7" strokeLinecap="round" />
        <path d="M32 61h30l-2 10a6 6 0 0 1-6 5H40a6 6 0 0 1-6-5l-2-10Z" fill="#E8A582" stroke="#50382E" strokeWidth="2.1" />
      </g>
      <g className="agent-head">
        {variant === 'fox' ? (
          <>
            <path d="M25 44 22 20l21 11h8l19-11-2 25c3 13-10 19-22 22-14-4-26-10-21-23Z" fill="#E28560" stroke="#50382E" strokeWidth="2.2" strokeLinejoin="round" />
            <path d="m26 26 4 12 8-5Zm40 0-4 12-8-5Z" fill="#FFC6A9" />
            <path d="M27 48c8-3 14 1 19 10 5-9 11-13 21-10-1 9-11 15-21 18-10-3-18-9-19-18Z" fill="#FFF3DF" />
          </>
        ) : variant === 'owl' ? (
          <>
            <path d="M22 38v-9l8 4c8-9 25-9 33 0l8-4v12c2 17-9 25-25 25S20 55 22 38Z" fill="#A89BCF" stroke="#40334E" strokeWidth="2.2" strokeLinejoin="round" />
            <path d="M27 45c0-11 14-14 19-5 5-9 19-6 19 5S54 62 46 62 27 54 27 45Z" fill="#F4E9FF" stroke="#71618F" strokeWidth="1.3" />
            <path d="m43 55 3 5 3-5" fill="#E1A05F" />
          </>
        ) : (
          <>
            <path d="M43 26c-1-7 1-10 5-13" stroke="#50382E" strokeWidth="2.5" strokeLinecap="round" />
            <circle cx="50" cy="12.5" r="3.2" fill="#D97757" stroke="#50382E" strokeWidth="1.7" />
            <path d="M24 38h-2a5 5 0 0 0-5 5v6a5 5 0 0 0 5 5h2m43-16h2a5 5 0 0 1 5 5v6a5 5 0 0 1-5 5h-2" fill="#E8A582" stroke="#50382E" strokeWidth="2.1" />
            <path d="M24 39c0-9 7-14 16-15l11-.3c11 0 18 6 18 15l-.6 13c-.5 8-6.5 13-14.3 13H39c-9.4 0-15.5-5-15.5-13.6Z" fill="#EAB390" stroke="#50382E" strokeWidth="2.2" />
            <path d="M28.3 42.4c.4-5 4.5-9.2 10.1-9.4h15.5c6.3 0 10.2 3.8 10.2 9.2l-.3 8.4c-.3 5.5-4.6 9.6-10.6 9.6H39.6c-6.6 0-11.2-3.6-11.5-9.1Z" fill="#FFF3DF" stroke="#704A3C" strokeWidth="1.55" />
            <path d="M32.2 42.5c.5-3.2 3.1-5.4 6.5-5.8H45" stroke="#FFFDF7" strokeWidth="2" strokeLinecap="round" />
          </>
        )}
        <g className="agent-eyes" stroke="#483228" strokeWidth="2.8" strokeLinecap="round">
          {state === 'done' ? <path d="M36 46q2.2-4 4.4 0m11 0q2.2-4 4.4 0" /> : <path d="M38.5 43.5v3.7m14.6-3.7v3.7" />}
        </g>
        <path d={alert ? 'M44 53q2-1.4 4 0' : state === 'thinking' ? 'M44.6 52.2a1.5 1.5 0 1 0 3 0a1.5 1.5 0 0 0-3 0' : 'M43.4 51.3q2.9 3.1 5.7 0'}
          stroke="#54392E" strokeWidth="1.7" strokeLinecap="round" />
        <path d="M32.7 50.5h2.9m21.2 0h2.8" stroke="#E8A582" strokeWidth="2" strokeLinecap="round" />
      </g>
      {!alert && (
        <g className="agent-quill" stroke="#50382E" strokeLinecap="round" strokeLinejoin="round">
          <path d="M66 70c2-12 7-25 20-29-.6 12-6 19-17 21" fill="#D97757" strokeWidth="1.8" />
          <path d="M67 68c4-8 9-14 14-19m-6 7.4 5.7-1.2" strokeWidth="1.5" />
          <path className="agent-ink" d="M69 77q7-3 14 0" strokeWidth="1.8" />
        </g>
      )}
      {alert && (
        <g>
          <circle cx="74.5" cy="66" r="13.5" fill="var(--surface)" />
          <circle cx="74.5" cy="66" r="11.2" fill="#F3D597" stroke="#855928" strokeWidth="1.5" />
          <path d="M74.5 60.7v5.7m0 4.1v.1" stroke="#72471C" strokeWidth="2.2" strokeLinecap="round" />
        </g>
      )}
    </svg>
  );
}

import React, { useEffect, useRef } from 'react';
import { Check, ChevronDown } from 'lucide-react';

export default function ModelPicker() {
  const picker = useRef(null);

  useEffect(() => {
    function closeOutside(event) {
      if (picker.current && !picker.current.contains(event.target)) picker.current.open = false;
    }

    document.addEventListener('pointerdown', closeOutside);
    return () => document.removeEventListener('pointerdown', closeOutside);
  }, []);

  return (
    <details className="model-picker" ref={picker} onKeyDown={(event) => {
      if (event.key === 'Escape') picker.current.open = false;
    }}>
      <summary aria-label="Active model, KAAL-304M">KAAL-304M<ChevronDown size={13} aria-hidden="true" /></summary>
      <ul className="model-options" aria-label="Models">
        <li><span>KAAL-304M</span><span className="current-model"><Check size={13} aria-hidden="true" /> Active</span></li>
      </ul>
    </details>
  );
}

import React, { useRef, useState } from 'react';
import { ArrowUpRight, Info, X } from 'lucide-react';

export default function ModelGuide({ compact }) {
  const [noticeVisible, setNoticeVisible] = useState(true);
  const details = useRef(null);
  const openDetails = () => details.current.showModal();

  return (
    <>
      {noticeVisible ? (
        <aside className="model-notice" data-compact={compact} aria-labelledby="model-notice-title">
          <Info size={21} aria-hidden="true" />
          <div className="model-notice-copy">
            <h2 id="model-notice-title">Read Before Use</h2>
            {compact ? (
              <p>KAAL-304M is an experimental checkpoint. Verify factual claims and calculations before relying on them.</p>
            ) : (
              <>
                <p>KAAL-304M is a 304-million-parameter language model trained from scratch with limited compute. The training budget constrained how long it could pretrain. It can produce fluent text while getting facts, arithmetic, or instructions wrong.</p>
                <p>In local testing, it sometimes repeated the prompt, described the topic instead of performing the task, or invented details. This checkpoint is available for local inference and experimentation. Extended pretraining is underway, with updated weights planned after further training and evaluation.</p>
              </>
            )}
            <button type="button" className="model-guide-link" onClick={openDetails}>
              Architecture, evaluation, and limitations <ArrowUpRight size={15} aria-hidden="true" />
            </button>
          </div>
          <button type="button" className="notice-close" aria-label="Dismiss model notice" onClick={() => setNoticeVisible(false)}>
            <X size={17} aria-hidden="true" />
          </button>
        </aside>
      ) : (
        <div className="model-notice-reopen">
          <button type="button" className="model-guide-link" onClick={() => setNoticeVisible(true)}>
            <Info size={14} aria-hidden="true" /> About this model
          </button>
        </div>
      )}

      <dialog className="research-dialog model-guide" ref={details} aria-labelledby="model-guide-title">
        <header className="research-bar">
          <strong>TRINETRA RESEARCH</strong>
          <form method="dialog">
            <button className="notice-close" aria-label="Close model details"><X size={18} aria-hidden="true" /></button>
          </form>
        </header>
        <div className="research-content">
          <p className="research-label">Model notes</p>
          <h2 id="model-guide-title">Inside KAAL-304M</h2>
          <p className="research-lead">An experimental decoder-only Transformer, trained from scratch and released for local inference and experimentation.</p>

          <dl className="research-grid model-specs">
            <div><dt>Parameters</dt><dd>304,355,328 unique</dd></div>
            <div><dt>Architecture</dt><dd>20 Transformer blocks, 16 attention heads per block</dd></div>
            <div><dt>Context window</dt><dd>2,048 tokens with learned position embeddings</dd></div>
            <div><dt>Generation limit</dt><dd>256 new tokens per response</dd></div>
          </dl>

          <h3>Why a fluent answer can be wrong</h3>
          <p>The decoder generates a response one token at a time from the prompt and the text it has already written. The chat pipeline has no retrieval or fact-checking step, so a likely-looking continuation can include an invented detail or a wrong calculation. The parameter count describes the model’s size. It does not measure how well the checkpoint learned a particular skill.</p>

          <h3>What limited compute meant for this checkpoint</h3>
          <p>The available compute constrained how long the model could pretrain. That is a credible contributor to weak factual recall and reasoning. The inference code and these tests cannot tell us how much of the behavior comes from model size, training data, pretraining duration, or instruction training. Isolating those effects would require controlled comparisons across checkpoints.</p>

          <h3>What happened in local testing</h3>
          <p>Earlier spot checks found factual and arithmetic errors. A later run sent 1,000 distinct prompts in sequence across creative writing, description, brainstorming, rewriting, and summarization. All 1,000 streams completed. Many replies stayed on topic but missed the task. For example, when asked to rewrite “I paint a lighthouse every summer” in the past tense, the model returned the original sentence. A broken-clock story prompt produced a time-machine premise on one run and an unrelated account of clock history on replay.</p>
          <p>The 1,000-prompt run was exploratory. Stream completion measures whether inference finished, and the handpicked starter prompts show examples of what the model can sometimes do. Neither establishes an accuracy rate.</p>

          <h3>Why the same prompt can give a different answer</h3>
          <p>The engine samples from likely next tokens using a fixed decoding profile. Sampling can change the wording and sometimes the direction of a reply. The context window is also finite. The engine reserves room for the response and drops the beginning of overlong inputs. That can remove useful instructions or reference text, although it does not explain failures on the short prompts above.</p>

          <h3>Using this checkpoint</h3>
          <p>Try short experiments with one clear task. Use Add Reference Context when there is a passage the model should work from, and check the reply against that source. Extended pretraining is underway. Updated weights are planned after additional training and evaluation.</p>
        </div>
      </dialog>
    </>
  );
}

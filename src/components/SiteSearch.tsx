import { useEffect, useMemo, useRef, useState } from 'react';
import {
  searchSite,
  splitHighlightSegments,
  type SearchKind,
} from '../content/search';
import { T } from '../i18n';
import { Tx } from '../i18n/jsx';
import '../styles/search.css';

type FilterKind = SearchKind | 'all';

const KIND_LABELS: Record<SearchKind, string> = {
  experiment: T('实验'),
  note: T('笔记'),
};

const QUICK_QUERIES = ['Attention', 'RoPE', 'KV Cache', 'Flash', 'MoE', 'AWQ', 'GRPO', 'model.py'];

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable;
}

export function SiteSearch() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [kind, setKind] = useState<FilterKind>('all');
  const [activeIndex, setActiveIndex] = useState(0);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  const hits = useMemo(() => searchSite(query, kind, 16), [query, kind]);
  const safeIndex = hits.length ? Math.min(activeIndex, hits.length - 1) : 0;

  const close = () => {
    setOpen(false);
    requestAnimationFrame(() => triggerRef.current?.focus({ preventScroll: true }));
  };

  useEffect(() => {
    const onGlobalKey = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        setOpen((prev) => !prev);
        return;
      }
      if (!open && event.key === '/' && !event.ctrlKey && !event.metaKey && !event.altKey) {
        if (!isEditableTarget(event.target)) {
          event.preventDefault();
          setOpen(true);
        }
      }
    };
    window.addEventListener('keydown', onGlobalKey);
    return () => window.removeEventListener('keydown', onGlobalKey);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    requestAnimationFrame(() => inputRef.current?.focus());
    return () => {
      document.body.style.overflow = prevOverflow;
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const activeEl = listRef.current?.querySelector<HTMLElement>('[data-active="true"]');
    activeEl?.scrollIntoView({ block: 'nearest' });
  }, [open, safeIndex]);

  function onInputKeyDown(event: React.KeyboardEvent<HTMLInputElement>) {
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      close();
    } else if (event.key === 'ArrowDown') {
      event.preventDefault();
      if (hits.length) setActiveIndex((safeIndex + 1) % hits.length);
    } else if (event.key === 'ArrowUp') {
      event.preventDefault();
      if (hits.length) setActiveIndex((safeIndex - 1 + hits.length) % hits.length);
    } else if (event.key === 'Enter' && hits[safeIndex]) {
      event.preventDefault();
      const targetHref = hits[safeIndex].item.href;
      setOpen(false);
      window.location.assign(targetHref);
    }
  }

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        className="site-search-trigger"
        aria-label={T('站内搜索')}
        title={T('站内搜索（Ctrl+K 或 /）')}
        aria-haspopup="dialog"
        aria-expanded={open}
        onClick={() => setOpen(true)}
      >
        <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
          <circle cx="8.8" cy="8.8" r="5.6" />
          <path d="M13 13l4.2 4.2" />
        </svg>
        <span className="site-search-label">{T('搜索')}</span>
        <kbd className="site-search-kbd" aria-hidden="true">
          Ctrl K
        </kbd>
      </button>

      {open && (
        <div
          className="site-search-backdrop"
          onClick={(event) => {
            if (event.target === event.currentTarget) close();
          }}
        >
          <div
            className="site-search-dialog"
            role="dialog"
            aria-modal="true"
            aria-label={T('站内搜索')}
            onKeyDown={(event) => {
              if (event.key === 'Escape') {
                event.preventDefault();
                event.stopPropagation();
                close();
              }
            }}
          >
            <div className="site-search-head">
              <svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true">
                <circle cx="8.8" cy="8.8" r="5.6" />
                <path d="M13 13l4.2 4.2" />
              </svg>
              <input
                ref={inputRef}
                type="search"
                value={query}
                aria-label={T('搜索实验或概念')}
                placeholder={T('支持模糊搜索：实验、笔记、缩写（如 RoPE、GQA、AWQ）…')}
                onChange={(event) => {
                  setQuery(event.target.value);
                  setActiveIndex(0);
                }}
                onKeyDown={onInputKeyDown}
              />
              {query && (
                <button
                  type="button"
                  className="site-search-clear"
                  aria-label={T('清空搜索')}
                  onClick={() => {
                    setQuery('');
                    setActiveIndex(0);
                    inputRef.current?.focus();
                  }}
                >
                  ×
                </button>
              )}
              <button
                type="button"
                className="site-search-close"
                aria-label={T('关闭搜索')}
                onClick={close}
              >
                Esc
              </button>
            </div>

            <div className="site-search-filters" role="group" aria-label={T('搜索范围')}>
              {(
                [
                  ['all', T('全部')],
                  ['experiment', T('实验')],
                  ['note', T('知识笔记')],
                ] as const
              ).map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  aria-pressed={kind === value}
                  onClick={() => {
                    setKind(value);
                    setActiveIndex(0);
                    inputRef.current?.focus();
                  }}
                >
                  {label}
                </button>
              ))}
              <span className="site-search-count" role="status">
                {Tx('{#0} 条结果', hits.length)}
              </span>
            </div>

            {!query.trim() && (
              <div className="site-search-quick" aria-label={T('常用搜索词')}>
                <span>{T('试试搜：')}</span>
                {QUICK_QUERIES.map((term) => (
                  <button
                    key={term}
                    type="button"
                    onClick={() => {
                      setQuery(term);
                      setActiveIndex(0);
                      inputRef.current?.focus();
                    }}
                  >
                    {term}
                  </button>
                ))}
              </div>
            )}

            <ul
              ref={listRef}
              className="site-search-results"
              role="listbox"
              aria-label={T('搜索结果')}
            >
              {hits.map((hit, index) => {
                const isActive = index === safeIndex;
                const segments = splitHighlightSegments(hit.item.title, hit.titleRanges);
                return (
                  <li
                    key={hit.item.id}
                    role="option"
                    aria-selected={isActive}
                    data-active={isActive ? 'true' : undefined}
                  >
                    <a
                      href={hit.item.href}
                      onMouseEnter={() => setActiveIndex(index)}
                      onClick={() => setOpen(false)}
                    >
                      <span className="site-search-badge">{hit.item.badge}</span>
                      <span className="site-search-text">
                        <strong>
                          {segments.map((seg, i) =>
                            seg.hit ? <mark key={i}>{seg.text}</mark> : <span key={i}>{seg.text}</span>,
                          )}
                        </strong>
                        <small>{hit.item.subtitle}</small>
                      </span>
                      <span className={`site-search-kind kind-${hit.item.kind}`}>
                        {KIND_LABELS[hit.item.kind]}
                      </span>
                    </a>
                  </li>
                );
              })}
              {hits.length === 0 && (
                <li className="site-search-empty">
                  {T('没有找到匹配项，试试更短的关键词或英文缩写（如 RoPE、GQA、AWQ）。')}
                </li>
              )}
            </ul>

            <div className="site-search-foot">
              <span>{T('↑↓ 选择 · Enter 跳转 · Esc 关闭')}</span>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

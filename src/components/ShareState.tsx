import { useEffect, useState } from 'react';
import { T } from '../i18n';

/** Share only the explicitly serialized public experiment state, without URL query parameters. */
export function ShareState({ getHash }: { getHash: () => string }) {
  const [link, setLink] = useState('');
  const [notice, setNotice] = useState('');
  const serialized = getHash();
  useEffect(() => {
    setLink('');
    setNotice('');
  }, [serialized]);
  async function copy() {
    const hash = getHash();
    const url = window.location.origin + window.location.pathname + hash;
    window.history.replaceState(null, '', hash);
    try {
      await navigator.clipboard.writeText(url);
      setLink('');
      setNotice(T('已复制当前实验链接。'));
    } catch {
      setLink(url);
      setNotice(T('请选择下方链接并复制。'));
    }
  }
  return (
    <div className="chapter-share">
      <button className="button secondary" onClick={() => void copy()}>
        {T('复制实验链接')}
      </button>
      {notice && <p role="status">{notice}</p>}
      {link && (
        <input
          aria-label={T('可复制的实验链接')}
          readOnly
          value={link}
          onFocus={(e) => e.target.select()}
        />
      )}
    </div>
  );
}

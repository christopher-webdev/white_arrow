import { writable } from 'svelte/store';

type Settings = {
  threshold: number;        // 0..1
  includeReject: boolean;   // include meta_class=0
  onlyClasses: string;      // e.g. "1,2"
};

// load from localStorage
function load(): Settings {
  try {
    const raw = localStorage.getItem('wa_settings');
    if (!raw) return { threshold: 0, includeReject: false, onlyClasses: '' };
    const s = JSON.parse(raw);
    return {
      threshold: Number(s.threshold ?? 0),
      includeReject: Boolean(s.includeReject ?? false),
      onlyClasses: String(s.onlyClasses ?? '')
    };
  } catch {
    return { threshold: 0, includeReject: false, onlyClasses: '' };
  }
}

export const settings = writable<Settings>(load());

// persist on change
settings.subscribe((v) => {
  try { localStorage.setItem('wa_settings', JSON.stringify(v)); } catch {}
});

// build query string for API
export function buildQuery(s: Settings) {
  const p = new URLSearchParams();
  if (s.threshold && s.threshold > 0) p.set('threshold', String(s.threshold));
  if (s.includeReject) p.set('include_reject', 'true');
  if (s.onlyClasses.trim()) p.set('only_classes', s.onlyClasses.trim());
  const q = p.toString();
  return q ? `?${q}` : '';
}

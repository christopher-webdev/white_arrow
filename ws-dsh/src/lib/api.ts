import { PUBLIC_API_BASE, PUBLIC_API_TOKEN } from '$env/static/public';

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${PUBLIC_API_BASE}${path}`, {
    headers: PUBLIC_API_TOKEN ? { Authorization: `Bearer ${PUBLIC_API_TOKEN}` } : undefined
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText}${text ? `: ${text}` : ''}`);
  }
  return res.json();
}

export const api = {
  full: (q = '') => get(`/metrics/full${q}`),
  health: () => get<{ status: string }>('/health'),
  symbols: () => get<{ symbols: string[] }>('/symbols'),
  predictions: (symbol: string, limit = 200) =>
    get<any[]>(`/predictions/${encodeURIComponent(symbol)}?limit=${limit}`)
};

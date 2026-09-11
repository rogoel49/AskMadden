/* Phase 5.4: minimal service worker for Ask Madden.
 *
 * Scope is wherever this file is served from (/ui/ under web/dev_server.py),
 * so it only ever sees the frontend's own requests; /api/ lives outside the
 * scope and is never touched.
 *
 * Strategy, kept deliberately small:
 *   - the static assets (manifest, icons, Google Fonts CSS + font files):
 *     cache-first -- they change rarely and this is what makes the app
 *     shell paint instantly on launch from the Home Screen.
 *   - the HTML document itself: network-first, falling back to the cached
 *     copy. Cache-first here would mean an edit to the mockup does not show
 *     up on reload until the worker updates, which is hostile to the Phase
 *     5.5 landing-page work happening in that same file. The cached copy
 *     is only served when the network is down.
 *   - anything else (cross-origin, non-GET): not intercepted at all.
 *
 * Bump CACHE_NAME whenever the precache list changes; activate() drops the
 * old caches. No offline-first ambition: if the API is unreachable the page
 * still says so itself (its fetch() wrapper already handles that).
 */
const CACHE_NAME = 'askmadden-static-v1';
const PRECACHE = [
  'askmadden-ui-mockup.html',
  'manifest.json',
  'icons/icon-192.png',
  'icons/icon-512.png',
  'icons/apple-touch-icon.png',
];
const FONT_ORIGINS = ['https://fonts.googleapis.com', 'https://fonts.gstatic.com'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(PRECACHE.map((p) => new URL(p, self.location.href).href)))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  const inScope = url.origin === self.location.origin && url.pathname.startsWith(new URL(self.registration.scope).pathname);
  const isFont = FONT_ORIGINS.includes(url.origin);
  if (!inScope && !isFont) return;  // e.g. /api/* -- straight to the network, untouched

  if (req.mode === 'navigate' || url.pathname.endsWith('.html')) {
    event.respondWith(networkFirst(req));
  } else {
    event.respondWith(cacheFirst(req));
  }
});

async function cacheFirst(req) {
  const cache = await caches.open(CACHE_NAME);
  const hit = await cache.match(req);
  if (hit) return hit;
  const resp = await fetch(req);
  if (resp && resp.ok) cache.put(req, resp.clone());
  return resp;
}

async function networkFirst(req) {
  const cache = await caches.open(CACHE_NAME);
  try {
    const resp = await fetch(req);
    if (resp && resp.ok) cache.put(req, resp.clone());
    return resp;
  } catch (err) {
    const hit = await cache.match(req);
    if (hit) return hit;
    throw err;
  }
}

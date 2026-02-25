/**
 * antaris-flame — OpenClaw memory plugin (Flame tier)
 *
 * Pure JavaScript. Zero Python. Zero subprocesses. Zero bridge.
 * Memory cap: 5,000 | Lifespan: 100 days | BM25 + co-occurrence + SESSION_START bootstrap
 *
 * Hooks (ForgeMemoryPlugin equivalent):
 *   session_start       → bootstrap: load recent memories into buffer (void — OpenClaw ignores return)
 *   before_agent_start  → BEFORE_LLM enrichment: inject buffered session context + per-turn BM25 recall
 *   agent_end           → SESSION_END ingest: save new memories from conversation
 */

import fs   from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

// ── Tier constants ────────────────────────────────────────────────────────────
const TIER            = 'antaris-flame';
const MEMORY_CAP      = 5000;
const LIFESPAN_MS     = 100 * 24 * 60 * 60 * 1000; // 100 days
const RECALL_LIMIT    = 7;
const SESSION_LIMIT   = 5;  // memories injected at session_start
const MIN_RELEVANCE   = 0.08;
const COOC_WEIGHT     = 0.3;

// ── In-memory session bootstrap buffer ───────────────────────────────────────
// session_start is a VOID hook in OpenClaw — return values ignored.
// We buffer recall here and drain it on the first before_agent_start.
const sessionBuffer = new Map(); // sessionKey → { memories: [], used: false }

// ── BM25 search ───────────────────────────────────────────────────────────────
function tokenize(text) {
  return String(text).toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(t => t.length > 1);
}

function bm25(query, docs, k1 = 1.5, b = 0.75) {
  const qTokens = new Set(tokenize(query));
  if (!qTokens.size) return docs.map(() => 0);

  const avgLen = docs.reduce((s, d) => s + tokenize(d).length, 0) / Math.max(docs.length, 1);
  const N = docs.length;
  const df = {};
  for (const doc of docs) {
    for (const t of new Set(tokenize(doc))) df[t] = (df[t] || 0) + 1;
  }

  return docs.map(doc => {
    const tokens = tokenize(doc);
    const tf = {};
    for (const t of tokens) tf[t] = (tf[t] || 0) + 1;
    const dl = tokens.length;
    let score = 0;
    for (const qt of qTokens) {
      const f = tf[qt] || 0;
      if (!f) continue;
      const idf = Math.log((N - (df[qt] || 0) + 0.5) / ((df[qt] || 0) + 0.5) + 1);
      const tfn = (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgLen));
      score += idf * tfn;
    }
    return score;
  });
}

// ── Co-occurrence boost ───────────────────────────────────────────────────────
function buildCoocIndex(mems) {
  const cooc = {};
  for (const m of mems) {
    const arr = [...new Set(tokenize(m.text))];
    for (let i = 0; i < arr.length; i++) {
      for (let j = i + 1; j < Math.min(arr.length, i + 5); j++) {
        const key = arr[i] < arr[j] ? `${arr[i]}|${arr[j]}` : `${arr[j]}|${arr[i]}`;
        cooc[key] = (cooc[key] || 0) + 1;
      }
    }
  }
  return cooc;
}

function coocBoost(query, docText, coocIndex) {
  const qToks = [...new Set(tokenize(query))];
  const dToks = new Set(tokenize(docText));
  let boost = 0;
  for (let i = 0; i < qToks.length; i++) {
    for (let j = i + 1; j < Math.min(qToks.length, i + 5); j++) {
      const key = qToks[i] < qToks[j] ? `${qToks[i]}|${qToks[j]}` : `${qToks[j]}|${qToks[i]}`;
      if (coocIndex[key] && (dToks.has(qToks[i]) || dToks.has(qToks[j]))) {
        boost += Math.min(coocIndex[key] / 10, 0.5);
      }
    }
  }
  return Math.min(boost, COOC_WEIGHT);
}

// ── Memory store I/O ──────────────────────────────────────────────────────────
function storePath(memDir) { return path.join(memDir, 'memories.json'); }

function loadMemories(memDir) {
  try {
    fs.mkdirSync(memDir, { recursive: true });
    const p = storePath(memDir);
    if (!fs.existsSync(p)) return [];
    return JSON.parse(fs.readFileSync(p, 'utf8'));
  } catch { return []; }
}

function saveMemories(memDir, mems) {
  try {
    fs.mkdirSync(memDir, { recursive: true });
    fs.writeFileSync(storePath(memDir), JSON.stringify(mems, null, 2));
  } catch { /* non-fatal */ }
}

function pruneExpired(mems) {
  return mems.filter(m => m.ts > Date.now() - LIFESPAN_MS);
}

function enforceCap(mems) {
  return mems.length <= MEMORY_CAP ? mems : mems.slice(mems.length - MEMORY_CAP);
}

// ── Text extraction helpers ───────────────────────────────────────────────────
function lastUserMessage(messages) {
  if (!Array.isArray(messages)) return '';
  const userMsgs = messages.filter(m => m.role === 'user');
  if (!userMsgs.length) return '';
  const last = userMsgs[userMsgs.length - 1];
  const c = Array.isArray(last.content) ? last.content.map(x => x.text || '').join(' ') : String(last.content || '');
  return c.trim().slice(0, 1000);
}

function lastAssistantMessage(messages) {
  if (!Array.isArray(messages)) return '';
  const msgs = messages.filter(m => m.role === 'assistant');
  if (!msgs.length) return '';
  const last = msgs[msgs.length - 1];
  const c = Array.isArray(last.content) ? last.content.map(x => x.text || '').join(' ') : String(last.content || '');
  return c.trim().slice(0, 500);
}

function resolveMemDir(config, event) {
  const base = config.memoryPath || path.join(process.env.HOME || '~', '.openclaw', `${TIER}-store`);
  const key  = event?.sessionKey || event?.sessionId || event?.session?.key || event?.channelId || 'default';
  return path.join(base, String(key).replace(/[^a-zA-Z0-9_-]/g, '_'));
}

function sessionKey(event) {
  return event?.sessionKey || event?.sessionId || event?.session?.key || event?.channelId || 'default';
}

// ── Plugin entry point ────────────────────────────────────────────────────────
export default function plugin(api) {
  const config = api.config.get(TIER) || {};
  const log    = api.logger;

  log.info(`[${TIER}] Loaded — cap: ${MEMORY_CAP} | lifespan: 100 days | BM25 + co-occurrence | SESSION_START bootstrap`);

  // ── SESSION_START: bootstrap buffer ─────────────────────────────────────────
  // NOTE: OpenClaw ignores return values from session_start.
  // We buffer the top memories here and drain them on the first before_agent_start.
  api.on('session_start', async (event) => {
    try {
      const memDir  = resolveMemDir(config, event);
      const sKey    = sessionKey(event);
      const raw     = loadMemories(memDir);
      const live    = pruneExpired(raw);
      if (!live.length) return;

      // Bootstrap: take the N most recent memories to orient the session
      const recent = [...live].sort((a, b) => b.ts - a.ts).slice(0, SESSION_LIMIT);
      sessionBuffer.set(sKey, { memories: recent, used: false });

      log.info(`[${TIER}] session_start: buffered ${recent.length} memories for session ${sKey}`);
    } catch (err) {
      log.warn(`[${TIER}] session_start error: ${err.message}`);
    }
  });

  // ── BEFORE_LLM: enrichment (before_agent_start) ──────────────────────────
  api.on('before_agent_start', async (event) => {
    try {
      const memDir = resolveMemDir(config, event);
      const sKey   = sessionKey(event);
      const query  = lastUserMessage(event.messages);
      if (!query) return {};

      const raw    = loadMemories(memDir);
      const live   = pruneExpired(raw);
      const cooc   = buildCoocIndex(live);

      // Per-turn BM25 + co-occurrence recall
      const texts  = live.map(m => m.text);
      const scores = bm25(query, texts);
      const ranked = live
        .map((m, i) => ({ m, score: scores[i] + coocBoost(query, m.text, cooc) }))
        .filter(x => x.score >= MIN_RELEVANCE)
        .sort((a, b) => b.score - a.score)
        .slice(0, RECALL_LIMIT)
        .map(x => x.m);

      // Drain session_start buffer on first turn
      const buf    = sessionBuffer.get(sKey);
      const parts  = [];

      if (buf && !buf.used) {
        buf.used = true;
        sessionBuffer.set(sKey, buf);
        const sessionMems = buf.memories.filter(m => !ranked.find(r => r.id === m.id));
        if (sessionMems.length) {
          parts.push(`[${TIER.toUpperCase()} SESSION CONTEXT — ${sessionMems.length} recent]`);
          sessionMems.forEach((m, i) => parts.push(`${i + 1}. ${m.text}`));
          parts.push('');
        }
      }

      if (ranked.length) {
        parts.push(`[${TIER.toUpperCase()} MEMORY — ${ranked.length} relevant]`);
        ranked.forEach((m, i) => parts.push(`${i + 1}. ${m.text}`));
        parts.push('');
      }

      return parts.length ? { prependContext: parts.join('\n') } : {};
    } catch (err) {
      log.warn(`[${TIER}] before_agent_start error: ${err.message}`);
      return {};
    }
  });

  // ── SESSION_END: ingest (agent_end) ──────────────────────────────────────
  api.on('agent_end', async (event) => {
    try {
      const memDir   = resolveMemDir(config, event);
      const userText = lastUserMessage(event.messages);
      const botText  = lastAssistantMessage(event.messages);
      if (!userText) return;

      const raw  = loadMemories(memDir);
      const live = pruneExpired(raw);

      const newMem = {
        id:     crypto.randomUUID(),
        text:   `User: ${userText}${botText ? `\nBot: ${botText}` : ''}`,
        ts:     Date.now(),
        source: 'conversation',
      };

      saveMemories(memDir, enforceCap([...live, newMem]));
    } catch (err) {
      log.warn(`[${TIER}] agent_end error: ${err.message}`);
    }
  });
}

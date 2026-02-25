/**
 * antaris-flame — OpenClaw memory plugin (Flame tier)
 *
 * Pure JavaScript. Zero Python. Zero subprocesses. Zero bridge.
 * Memory cap: 5,000 | Lifespan: 100 days | Search: BM25 + co-occurrence boost
 *
 * Hooks:
 *   before_agent_start  → recall relevant memories, inject into context
 *   agent_end           → ingest new memories from conversation
 */

import fs   from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

// ── Tier constants ────────────────────────────────────────────────────────────
const TIER        = 'antaris-flame';
const MEMORY_CAP  = 5000;
const LIFESPAN_MS = 100 * 24 * 60 * 60 * 1000; // 100 days
const RECALL_LIMIT = 7;
const MIN_RELEVANCE = 0.08;
const COOC_WEIGHT  = 0.3; // additive boost from co-occurrence

// ── BM25 search ───────────────────────────────────────────────────────────────
function tokenize(text) {
  return String(text).toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(t => t.length > 1);
}

function bm25(query, docs, k1 = 1.5, b = 0.75) {
  const qTokens = new Set(tokenize(query));
  if (qTokens.size === 0) return docs.map(() => 0);

  const avgLen = docs.reduce((s, d) => s + tokenize(d).length, 0) / Math.max(docs.length, 1);
  const N = docs.length;

  const df = {};
  for (const doc of docs) {
    const seen = new Set(tokenize(doc));
    for (const t of seen) df[t] = (df[t] || 0) + 1;
  }

  return docs.map(doc => {
    const tokens = tokenize(doc);
    const tf = {};
    for (const t of tokens) tf[t] = (tf[t] || 0) + 1;
    const dl = tokens.length;

    let score = 0;
    for (const qt of qTokens) {
      const f  = tf[qt] || 0;
      if (f === 0) continue;
      const idf = Math.log((N - (df[qt] || 0) + 0.5) / ((df[qt] || 0) + 0.5) + 1);
      const tfn = (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgLen));
      score += idf * tfn;
    }
    return score;
  });
}

// ── Co-occurrence index ───────────────────────────────────────────────────────
// Tracks word pairs that frequently appear together across memories.
// At recall time, adds a small relevance boost for memories that share
// common term pairs with the query.
function buildCoocIndex(mems) {
  const cooc = {};
  for (const m of mems) {
    const toks = tokenize(m.text);
    const seen = new Set(toks);
    const arr  = [...seen];
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
  const cutoff = Date.now() - LIFESPAN_MS;
  return mems.filter(m => m.ts > cutoff);
}

function enforceCap(mems) {
  if (mems.length <= MEMORY_CAP) return mems;
  return mems.slice(mems.length - MEMORY_CAP);
}

// ── Text extraction helpers ───────────────────────────────────────────────────
function lastUserMessage(messages) {
  if (!Array.isArray(messages)) return '';
  const userMsgs = messages.filter(m => m.role === 'user');
  if (!userMsgs.length) return '';
  const last = userMsgs[userMsgs.length - 1];
  const content = Array.isArray(last.content) ? last.content.map(c => c.text || '').join(' ') : String(last.content || '');
  return content.trim().slice(0, 1000);
}

function lastAssistantMessage(messages) {
  if (!Array.isArray(messages)) return '';
  const msgs = messages.filter(m => m.role === 'assistant');
  if (!msgs.length) return '';
  const last = msgs[msgs.length - 1];
  const content = Array.isArray(last.content) ? last.content.map(c => c.text || '').join(' ') : String(last.content || '');
  return content.trim().slice(0, 500);
}

function resolveMemDir(config, event) {
  const base = config.memoryPath || path.join(process.env.HOME || '~', '.openclaw', `${TIER}-store`);
  const sessionKey = event?.sessionKey || event?.sessionId || event?.session?.key || event?.channelId || 'default';
  return path.join(base, String(sessionKey).replace(/[^a-zA-Z0-9_-]/g, '_'));
}

// ── Plugin entry point ────────────────────────────────────────────────────────
export default function plugin(api) {
  const config = api.config.get(TIER) || {};
  const log    = api.logger;

  log.info(`[${TIER}] Loaded — cap: ${MEMORY_CAP} memories | lifespan: 100 days | BM25 + co-occurrence`);

  // ── Recall: before_agent_start ─────────────────────────────────────────────
  api.on('before_agent_start', async (event) => {
    try {
      const memDir = resolveMemDir(config, event);
      const query  = lastUserMessage(event.messages);
      if (!query) return {};

      const raw  = loadMemories(memDir);
      const live = pruneExpired(raw);
      if (!live.length) return {};

      const texts  = live.map(m => m.text);
      const scores = bm25(query, texts);
      const cooc   = buildCoocIndex(live);

      const ranked = live
        .map((m, i) => ({ m, score: scores[i] + coocBoost(query, m.text, cooc) }))
        .filter(x => x.score >= MIN_RELEVANCE)
        .sort((a, b) => b.score - a.score)
        .slice(0, RECALL_LIMIT)
        .map(x => x.m);

      if (!ranked.length) return {};

      const block = [
        `[${TIER.toUpperCase()} MEMORY — ${ranked.length} recall(s)]`,
        ...ranked.map((m, i) => `${i + 1}. ${m.text}`),
        '',
      ].join('\n');

      return { prependContext: block };
    } catch (err) {
      log.warn(`[${TIER}] recall error: ${err.message}`);
      return {};
    }
  });

  // ── Ingest: agent_end ──────────────────────────────────────────────────────
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
      log.warn(`[${TIER}] ingest error: ${err.message}`);
    }
  });
}

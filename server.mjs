import express from "express";
import fetch from "node-fetch";
import crypto from "crypto";
import { z } from "zod";

const app = express();

// static site (serves your index.html)
app.use(express.static(".", { index: "index.html" }));

// basic JSON body and CORS
app.use(express.json({ limit: "2mb" }));
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
  next();
});

// tiny cache
const CACHE = new Map();
const cacheKey = (x) => crypto.createHash("sha256").update(JSON.stringify(x)).digest("hex");
const getCache = (k) => {
  const e = CACHE.get(k);
  if (!e) return;
  if (Date.now() > e.t + 10 * 60 * 1000) { CACHE.delete(k); return; } // 10 min TTL
  return e.v;
};
const setCache = (k, v) => CACHE.set(k, { t: Date.now(), v });

app.get("/health", (_req, res) => res.json({ ok: true, message: "Nexus Relay is active" }));

app.post("/chat", async (req, res) => {
  try {
    const schema = z.object({
      model: z.string(),
      prompt: z.string(),
      params: z.object({
        temperature: z.number().optional(),
        max_new_tokens: z.number().optional()
      }).optional()
    });

    const body = schema.parse(req.body);
    const k = cacheKey({ path: "/chat", body });
    const hit = getCache(k);
    if (hit) return res.json(hit);

    const hfToken = process.env.HF_TOKEN || "";
    const resp = await fetch(`https://api-inference.huggingface.co/models/${encodeURIComponent(body.model)}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "authorization": `Bearer ${hfToken}`
      },
      body: JSON.stringify({
        inputs: body.prompt,
        parameters: {
          temperature: body.params?.temperature ?? 0.8,
          max_new_tokens: body.params?.max_new_tokens ?? 256,
          return_full_text: false
        }
      })
    });

    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error(`Hugging Face API error: ${resp.status} - ${txt}`);
    }

    const j = await resp.json();
    const text = Array.isArray(j) ? (j[0]?.generated_text ?? "") : (j.generated_text ?? "");
    const out = { text };
    setCache(k, out);
    res.json(out);
  } catch (err) {
    console.error("Relay /chat error:", err);
    res.status(500).json({ error: String(err.message || err) });
  }
});

const port = process.env.PORT || 8081;
app.listen(port, () => console.log(`Nexus Relay + static site on :${port}`));

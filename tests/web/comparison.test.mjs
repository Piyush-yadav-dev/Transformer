import test from "node:test";
import assert from "node:assert/strict";
import { session_state } from "../../frontend/application.js";
const { createSession } = session_state;
import { comparison_page } from "../../frontend/application.js";
const { createComparison } = comparison_page;
import { generation_client } from "../../frontend/application.js";
const { requestFromDraft } = generation_client;
const catalog = { registry_fingerprint: "c".repeat(64), models: ["baseline", "combined"].map((kind,i) => ({
 model_id: `fixture-${kind}`, kind, display_name: `Synthetic ${kind}`, availability: "registered",
 unavailable_reason: null, checkpoint:{sha256:String(i+1).repeat(64),update:1},tokenizer_fingerprint:"b".repeat(64)
})) };
function ready() { const s=createSession();s.setCatalog(catalog);s.updateDraft({prompt:"A story.",mode:"sample",variation:"0.7",topK:"20",seed:"42",length:"32"});return s; }
function reply(r) {
 const m=catalog.models.find(m=>m.model_id===r.model_id);
 return {ok:true,json:async()=>({request_id:"d".repeat(32),prompt:r.prompt,continuation:"Test fixture.",model_id:m.model_id,model_kind:m.kind,
 checkpoint:m.checkpoint,tokenizer_fingerprint:m.tokenizer_fingerprint,registry_fingerprint:catalog.registry_fingerprint,
 sampling_settings:{...r.settings,temperature:r.settings.temperature??0,top_k:r.settings.top_k??null,seed:r.settings.seed??null,effective_seed:r.settings.seed??null,stop_at_eos:true},
 generated_token_count:10,stop_reason:"max_new_tokens",generation_duration_seconds:.1,model_load_seconds:.2,context_length:256,
 initial_prompt_tokens_truncated:0,unknown_prompt_tokens:0,device:"cpu",cache_hit:false})};
}
test("comparison sends identical settings sequentially; history retains identities through switching and reopening",async()=>{
 const s=ready(),requests=[];let active=0,peak=0;
 const c=createComparison(s,{fetcher:async(_,o)=>{active++;peak=Math.max(peak,active);const r=JSON.parse(o.body);requests.push(r);await new Promise(r=>setTimeout(r,2));active--;return reply(r);}});
 const pending=c.run();await c.run();await pending;
 assert.equal(peak,1);assert.equal(requests.length,2);
 assert.deepEqual(requests[0].settings,requests[1].settings);assert.equal(requests[0].prompt,requests[1].prompt);
 assert.equal(c.results.length,2);assert.equal(s.snapshot().history.length,2);
 const a=s.snapshot().history[1];s.restoreHistory(a.id);s.updateDraft({model:"fixture-combined"});
 assert.equal(s.snapshot().result.metadata.model_id,"fixture-baseline");
 const stop=s.subscribe(()=>{});stop();assert.equal(requests.length,2);
 s.clearHistory();assert.equal(s.snapshot().history.length,0);assert.equal(s.snapshot().result.metadata.model_id,"fixture-baseline");
});
test("blank comparison seed is generated once and recorded for both outputs",async()=>{
 const s=ready();s.updateDraft({seed:""});const calls=[];
 await createComparison(s,{fetcher:async(_,o)=>{const r=JSON.parse(o.body);calls.push(r);return reply(r);}}).run();
 assert.equal(calls.length,2);assert.match(calls[0].settings.seed,/^\d+$/);assert.equal(calls[0].settings.seed,calls[1].settings.seed);
});
test("missing comparison model makes no request and does not mark the other model unavailable",async()=>{
 const s=ready();s.setCatalog({...catalog,models:[catalog.models[0]]});let calls=0;
 await createComparison(s,{fetcher:async()=>{calls++;}}).run();
 assert.equal(calls,0);assert.match(s.snapshot().error.message,/both registered/);assert.equal(s.snapshot().models[0].availability,"registered");
});
test("uncertain lost response stops the pair without starting another job",async()=>{
 const s=ready();let calls=0;const c=createComparison(s,{fetcher:async()=>{calls++;throw Error("offline");}});
 await c.run();assert.equal(calls,1);assert.equal(c.results[0].state.error.code,"service_unavailable");assert.equal(c.running,false);
});
test("all controls map explicitly and greedy omits inactive sampling settings",()=>{
 const s=ready();assert.deepEqual(requestFromDraft(s.snapshot().draft).settings,{mode:"sample",max_new_tokens:32,unknown_policy:"error",temperature:.7,top_k:20,seed:"42"});
 for(const patch of [{length:"0"},{length:"1025"},{variation:"0"},{topK:"1.2"},{seed:"9223372036854775808"}]) assert.throws(()=>requestFromDraft({...s.snapshot().draft,...patch}));
 s.updateDraft({mode:"greedy"});assert.deepEqual(requestFromDraft(s.snapshot().draft).settings,{mode:"greedy",max_new_tokens:32,unknown_policy:"error"});
});

test("registry refresh never silently replaces an explicitly selected model",()=>{
 const s=ready();s.updateDraft({model:"fixture-combined"});
 s.setCatalog({...catalog,models:[catalog.models[0]]});
 assert.equal(s.snapshot().draft.model,"fixture-combined");
 assert.equal(s.snapshot().error.code,"model_not_found");
});

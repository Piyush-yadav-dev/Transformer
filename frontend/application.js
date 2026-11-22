// Website behavior grouped into named sections. No external JavaScript dependencies.

// Analytics Charts
const analytics_charts = (() => {
const ns = "http://www.w3.org/2000/svg";
function svg(tag, attrs = {}, text = "") {
  const element = document.createElementNS(ns, tag);
  for (const [key,value] of Object.entries(attrs)) element.setAttribute(key,String(value));
  element.textContent = text; return element;
}
function segments(points) {
  const groups = [];
  for (const point of points) {
    if (!groups.length || point.gap_before) groups.push([]);
    groups.at(-1).push(point);
  }
  return groups;
}
function chartDomain(series) {
  const points = series.flatMap(s => s.points);
  return { x: points.reduce((v,p)=>Math.max(v,p.tokens),1), y: points.reduce((v,p)=>Math.max(v,p.loss),1) * 1.05 };
}
function lossChart(title, series) {
  const section = document.createElement("section"); section.className = "analytics-chart";
  const heading = document.createElement("h2"); heading.textContent = title; section.append(heading);
  const available = series.filter(s => s.points.length);
  if (!available.length) { const p=document.createElement("p");p.textContent="No valid measurements available. Missing loss is not zero.";section.append(p);return section; }
  const legend=document.createElement("ul");legend.className="chart-legend";
  series.forEach((s,i)=>{const li=document.createElement("li");li.className=`series-${i%2}`;li.textContent=`${s.label} · ${s.points.length} raw measurements`;legend.append(li);});
  section.append(legend);
  const domain=chartDomain(series);
  const plot=svg("svg",{viewBox:"0 0 800 340",role:"img","aria-label":`${title}. Raw unsmoothed loss in nats per target token versus training target tokens processed. Exact values in the table below.`});
  const x=p=>70+690*p.tokens/domain.x, y=p=>285-235*p.loss/domain.y;
  for(let i=0;i<=4;i++){
    const yy=285-235*i/4, xx=70+690*i/4;
    plot.append(svg("line",{x1:70,x2:760,y1:yy,y2:yy,class:"chart-grid"}));
    plot.append(svg("text",{x:60,y:yy+5,"text-anchor":"end"},(domain.y*i/4).toFixed(2)));
    plot.append(svg("text",{x:xx,y:310,"text-anchor":"middle"},new Intl.NumberFormat("en",{notation:"compact",maximumFractionDigits:2}).format(domain.x*i/4)));
  }
  plot.append(svg("text",{x:70,y:25},"Loss · nats / target token"),svg("text",{x:415,y:335,"text-anchor":"middle"},"Training target tokens processed"));
  series.forEach((s,i)=>{
    for(const group of segments(s.points)) plot.append(svg("polyline",{points:group.map(p=>`${x(p)},${y(p)}`).join(" "),class:`chart-line series-${i%2}`}));
    // Har point raw log value hai; smoothing instability ko chhupata nahi.
    for(const p of s.points){
      const dot=svg("circle",{cx:x(p),cy:y(p),r:s.points.length>100?2:4,class:`chart-point series-${i%2}`});
      dot.append(svg("title",{},`${s.label} · update ${p.update} · ${p.tokens} tokens · ${p.loss} nats/token`));plot.append(dot);
    }
  });
  const scroll=document.createElement("div");scroll.className="chart-scroll";scroll.tabIndex=0;scroll.setAttribute("aria-label",title+" scrollable chart");scroll.append(plot);section.append(scroll);
  const help=document.createElement("p");help.className="field-help";help.textContent="Raw values, no smoothing. Scroll horizontally on narrow screens. Hover a point for details, or use the exact-value table. Training loss includes training-mode dropout; validation uses evaluation mode.";
  section.append(help);
  const details=document.createElement("details"),summary=document.createElement("summary");
  summary.textContent="Inspect raw measurements";details.append(summary);
  let initialized=false;
  details.addEventListener("toggle",()=>{if(!details.open||initialized)return;initialized=true;
    const rows=series.flatMap(s=>s.points.map(p=>[s.label,p.update,p.tokens,p.loss]));
    let page=0;const holder=document.createElement("div");holder.className="analytics-table";
    const nav=document.createElement("div");nav.className="table-controls";
    const prev=document.createElement("button"),next=document.createElement("button"),label=document.createElement("span");
    prev.textContent="Previous rows";next.textContent="Next rows";prev.type=next.type="button";prev.className=next.className="button button-secondary";
    function draw(){holder.replaceChildren();const table=document.createElement("table");
      const tr=document.createElement("tr");for(const name of ["Run","Update","Training target tokens","Loss (nats/token)"]){const th=document.createElement("th");th.scope="col";th.textContent=name;tr.append(th);}const head=document.createElement("thead");head.append(tr);table.append(head);
      const body=document.createElement("tbody");for(const row of rows.slice(page*50,page*50+50)){const tr=document.createElement("tr");for(const value of row){const td=document.createElement("td");td.textContent=String(value);tr.append(td);}body.append(tr);}table.append(body);holder.append(table);
      label.textContent=`Rows ${page*50+1}–${Math.min(rows.length,page*50+50)} of ${rows.length}`;prev.disabled=page===0;next.disabled=(page+1)*50>=rows.length;
    }
    prev.onclick=()=>{page--;draw();};next.onclick=()=>{page++;draw();};nav.append(prev,label,next);details.append(nav,holder);draw();
  });
  section.append(details);return section;
}
return { segments, chartDomain, lossChart };
})();

// Story Charts Data
const story_charts_data = (() => {
// BEGIN MEASURED CHART DATA
// Generated by tools/export_story_charts.py.
const chartData = {
  "model_id": "transformer-story-continuation-v2",
  "update": 96,
  "train": [
    {
      "update": 8,
      "loss": 2.1801,
      "gradient": 0.165
    },
    {
      "update": 16,
      "loss": 2.4766,
      "gradient": 0.247
    },
    {
      "update": 24,
      "loss": 1.9647,
      "gradient": 0.423
    },
    {
      "update": 32,
      "loss": 1.8197,
      "gradient": 0.367
    },
    {
      "update": 40,
      "loss": 2.4226,
      "gradient": 0.591
    },
    {
      "update": 48,
      "loss": 2.01,
      "gradient": 0.572
    },
    {
      "update": 56,
      "loss": 1.624,
      "gradient": 0.468
    },
    {
      "update": 64,
      "loss": 2.3563,
      "gradient": 0.763
    },
    {
      "update": 72,
      "loss": 2.2375,
      "gradient": 0.917
    },
    {
      "update": 80,
      "loss": 2.0322,
      "gradient": 0.815
    },
    {
      "update": 88,
      "loss": 2.2096,
      "gradient": 0.685
    },
    {
      "update": 96,
      "loss": 1.5824,
      "gradient": 0.619
    }
  ],
  "validation": [
    {
      "update": 0,
      "loss": 1.6765845268964767
    },
    {
      "update": 24,
      "loss": 1.6458
    },
    {
      "update": 48,
      "loss": 1.6114
    },
    {
      "update": 72,
      "loss": 1.5949
    },
    {
      "update": 96,
      "loss": 1.5852418690919876
    }
  ],
  "embedding": {
    "dimensions": 1536,
    "variance": [
      0.08590393353900852,
      0.07336333203216525
    ],
    "points": [
      {
        "word": "bird",
        "group": "Animals",
        "token_id": 11958,
        "x": -0.24145141677628898,
        "y": 0.02888164366587913
      },
      {
        "word": "cat",
        "group": "Animals",
        "token_id": 8251,
        "x": -0.6480891781628654,
        "y": -0.2088985307552969
      },
      {
        "word": "dog",
        "group": "Animals",
        "token_id": 5562,
        "x": -0.4454284180291202,
        "y": -0.08427657914098824
      },
      {
        "word": "fox",
        "group": "Animals",
        "token_id": 38835,
        "x": -0.2410844123230835,
        "y": -0.11715752181934995
      },
      {
        "word": "rabbit",
        "group": "Animals",
        "token_id": 38724,
        "x": -0.33689193181120164,
        "y": -0.10332534273486087
      },
      {
        "word": "kitten",
        "group": "Animals",
        "token_id": 76657,
        "x": -0.5012606440881909,
        "y": -0.2983202895577075
      },
      {
        "word": "garden",
        "group": "Places",
        "token_id": 13551,
        "x": 0.017340560888416484,
        "y": 0.22797559051395705
      },
      {
        "word": "school",
        "group": "Places",
        "token_id": 2906,
        "x": 0.15329607374753143,
        "y": 0.24563360214291938
      },
      {
        "word": "forest",
        "group": "Places",
        "token_id": 13638,
        "x": 0.027808274926472292,
        "y": 0.19205406196183797
      },
      {
        "word": "river",
        "group": "Places",
        "token_id": 14796,
        "x": -0.005456479962287485,
        "y": 0.15569265298158688
      },
      {
        "word": "library",
        "group": "Places",
        "token_id": 6733,
        "x": 0.109267054827794,
        "y": 0.28118452479945377
      },
      {
        "word": "village",
        "group": "Places",
        "token_id": 14126,
        "x": 0.15043624231908378,
        "y": 0.23078015198337046
      },
      {
        "word": "ribbon",
        "group": "Objects",
        "token_id": 44567,
        "x": 0.05012079061966221,
        "y": -0.021441730031867695
      },
      {
        "word": "key",
        "group": "Objects",
        "token_id": 1376,
        "x": 0.04350098215474331,
        "y": 0.2802150242282641
      },
      {
        "word": "box",
        "group": "Objects",
        "token_id": 3745,
        "x": 0.031821165298312123,
        "y": 0.34318774396359986
      },
      {
        "word": "boat",
        "group": "Objects",
        "token_id": 15328,
        "x": -0.061369781986068704,
        "y": 0.15045347984114552
      },
      {
        "word": "book",
        "group": "Objects",
        "token_id": 2311,
        "x": 0.09452836251881569,
        "y": 0.38771505990113403
      },
      {
        "word": "door",
        "group": "Objects",
        "token_id": 6006,
        "x": 0.03583011469093494,
        "y": 0.15717804416423692
      },
      {
        "word": "happy",
        "group": "Feelings",
        "token_id": 6247,
        "x": 0.3372927849491374,
        "y": -0.370497030585252
      },
      {
        "word": "sad",
        "group": "Feelings",
        "token_id": 12421,
        "x": 0.32153249425101715,
        "y": -0.27139999466288983
      },
      {
        "word": "afraid",
        "group": "Feelings",
        "token_id": 16575,
        "x": 0.28535164240650646,
        "y": -0.28791857257645537
      },
      {
        "word": "angry",
        "group": "Feelings",
        "token_id": 18514,
        "x": 0.31542829166391106,
        "y": -0.27419106901295026
      },
      {
        "word": "excited",
        "group": "Feelings",
        "token_id": 12035,
        "x": 0.2595410251617236,
        "y": -0.4082260527285349
      },
      {
        "word": "lonely",
        "group": "Feelings",
        "token_id": 39566,
        "x": 0.2479364027150459,
        "y": -0.23529886654123267
      }
    ]
  }
};
// END MEASURED CHART DATA
return { default: chartData };
})();

// Story Charts
const story_charts = (() => {
const data = story_charts_data.default;
const ns = "http://www.w3.org/2000/svg";

function dom(tag, text, className) {
  const element = document.createElement(tag);
  if (text) element.textContent = text;
  if (className) element.className = className;
  return element;
}

function svg(parent, tag, attrs, text) {
  const element = document.createElementNS(ns, tag);
  for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
  if (text) element.textContent = text;
  parent.append(element);
  return element;
}

function figure(title, description) {
  const card = dom("figure", null, "story-chart-card");
  const caption = dom("figcaption");
  caption.append(dom("h3", title), dom("p", description, "field-help"));
  card.append(caption);
  const graph = svg(card, "svg", { viewBox: "0 0 460 235", role: "img", "aria-label": title });
  return { card, graph };
}

function axis(graph, xLabel, yLabel) {
  svg(graph, "text", {x: 246, y: 228, "text-anchor": "middle"}, xLabel);
  svg(graph, "text", {x: 52, y: 16}, yLabel);
  svg(graph, "path", {d: "M52 30V190H438", class: "story-chart-axis"});
}

function trace(title, description, rows, key, unit) {
  const {card, graph} = figure(title, description);
  const maxUpdate = rows.at(-1).update;
  const values = rows.map(row => row[key]);
  const min = Math.floor(Math.min(...values) * 10) / 10;
  const max = Math.ceil(Math.max(...values) * 10) / 10;
  const x = row => 52 + 386 * row.update / maxUpdate;
  const y = row => 190 - 150 * (row[key] - min) / (max - min || 1);
  axis(graph, "Optimizer update", unit);
  for (let i = 0; i < 5; i++) {
    const yy = 190 - 150 * i / 4;
    svg(graph, "line", {x1: 52, x2: 438, y1: yy, y2: yy, class: "story-chart-gridline"});
    svg(graph, "text", {x: 45, y: yy + 4, "text-anchor": "end"}, (min + (max - min) * i / 4).toFixed(2));
    svg(graph, "text", {x: 52 + 386 * i / 4, y: 210, "text-anchor": "middle"}, String(Math.round(maxUpdate * i / 4)));
  }
  svg(graph, "polyline", {points: rows.map(row => `${x(row)},${y(row)}`).join(" "), class: "story-chart-line"});
  const details = dom("details");
  details.append(dom("summary", "View exact values"));
  const valuesList = dom("ul");
  for (const row of rows) {
    const point = svg(graph, "circle", {cx: x(row), cy: y(row), r: 4, class: "story-chart-point"});
    const fact = `Update ${row.update} · ${row[key].toFixed(4)}`;
    svg(point, "title", {}, fact);
    valuesList.append(dom("li", fact));
  }
  details.append(valuesList);
  card.append(details);
  return card;
}

function embeddingPlot() {
  const {points, dimensions, variance} = data.embedding;
  const {card, graph} = figure("Token embedding map",
    `A two-dimensional view of ${points.length} selected tokens from ${dimensions}-number input embeddings. The two axes capture ${Math.round(100 * (variance[0] + variance[1]))}% of the variation in these selected vectors.`);
  const colors = {Animals: "#245f56", Places: "#95601c", Objects: "#536097", Feelings: "#9a455d"};
  const xScale = Math.max(...points.map(point => Math.abs(point.x))) || 1;
  const yScale = Math.max(...points.map(point => Math.abs(point.y))) || 1;
  axis(graph, "Principal component 1", "Principal component 2");
  svg(graph, "path", {d: "M245 30V190M52 110H438", class: "story-chart-gridline"});
  const legend = dom("ul", null, "story-chart-legend");
  for (const [name, color] of Object.entries(colors)) {
    const item = dom("li");
    const dot = svg(item, "svg", {viewBox: "0 0 12 12", width: 12, height: 12, "aria-hidden": "true"});
    svg(dot, "circle", {cx: 6, cy: 6, r: 4, fill: color});
    item.append(document.createTextNode(name));
    legend.append(item);
  }
  const details = dom("details");
  details.append(dom("summary", "View plotted tokens"));
  const list = dom("ul");
  for (const point of points) {
    const {word, group, token_id, x, y} = point;
    const dot = svg(graph, "circle", {cx: 245 + 177*x/xScale, cy: 110 - 69*y/yScale, r: 6, fill: colors[group]});
    svg(dot, "title", {}, `${word} · ${group} · token ${token_id}`);
    list.append(dom("li", `${word} · ${group} · token ${token_id}`));
  }
  details.append(list);
  card.append(legend, dom("p", "Hover a dot to see its token. Colors mark hand-picked word groups. The embeddings are from the frozen base model. This projection shows only part of their structure.", "field-help"), details);
  return card;
}

function storyCharts(model) {
  const section = dom("section", null, "story-charts");
  section.append(dom("h2", "Training charts"));
  if (model.model_id !== data.model_id || model.checkpoint_update !== data.update) {
    section.append(dom("p", "Chart data is unavailable for this checkpoint.", "field-help"));
    return section;
  }
  const grid = dom("div", null, "story-chart-grid");
  grid.append(
    trace("Validation loss", "Average loss on eight separate examples. Update zero is the pretrained model before fine-tuning.", data.validation, "loss", "Loss · nats/token"),
    trace("Training token loss", "Mean continuation-token loss for the training example logged at every eighth update. Each point is one example.", data.train, "loss", "Loss · nats/token"),
    embeddingPlot(),
    trace("Gradient norm", "The logged size of the weight update signal before clipping. Spikes can help spot unstable training.", data.train, "gradient", "Gradient L2 norm"),
  );
  section.append(grid);
  return section;
}
return { storyCharts };
})();

// Insights Page
const insights_page = (() => {
const { storyCharts } = story_charts;
const { lossChart } = analytics_charts;
const fmt = value => value == null ? "Unavailable — no verified measurement" : typeof value === "number" ? value.toLocaleString(undefined,{maximumFractionDigits:4}) : String(value);
function el(tag,text,className) {const n=document.createElement(tag);if(text!=null)n.textContent=text;if(className)n.className=className;return n;}
function table(headers, rows) {
 const wrap=el("div",null,"analytics-table"), t=el("table"), head=el("thead"),tr=el("tr");
 for(const h of headers){const th=el("th",h);th.scope="col";tr.append(th);}head.append(tr);t.append(head);
 const body=el("tbody");for(const row of rows){const tr=el("tr");row.forEach((value,i)=>{const cell=el(i===0?"th":"td",fmt(value));if(i===0)cell.scope="row";if(typeof value==="number")cell.title=String(value);tr.append(cell);});body.append(tr);}t.append(body);wrap.append(t);return wrap;
}
function warnings(root, messages) { for(const message of messages) root.append(el("p",message,"inline-note")); }
function label(model){return `${model.kind==="baseline"?"Model A":"Model B"} · ${model.run_id}`;}
function overview(model) {
 const section=el("section",null,"model-overview");section.append(el("h2",label(model)),el("p",model.model_id,"field-help"));warnings(section,model.warnings);
 if(model.status!=="available")return section;
 const a=model.architecture;
 section.append(table(["Setting","Saved value"],[
 ["Layers",a.num_layers],["Attention heads",a.num_heads],["Embedding dimension",a.embedding_dim],
 ["Context length (tokens)",a.context_length],["Vocabulary size",a.vocab_size],
 ["Trainable parameters (shared weights counted once)",model.parameter_count],["Tokenizer",model.tokenizer_type],
 ["Training datasets",Object.entries(model.exposure).filter(([,n])=>n>0).map(([name])=>name).join(" + ")],
 ["Training budget (target tokens)",model.token_budget],["Optimizer-update budget",model.update_budget??"Token-budget run"],
 ["Checkpoint selection",model.selection_rule],["Selected update",model.checkpoint_update],
 ["Token/output weights tied",a.tie_embeddings?"Yes":"No"]
 ]));
 if(model.manifest){
  section.append(el("p",`Vocabulary: ${model.manifest.vocabulary_policy}`,"field-help"));
  section.append(el("p",`Training boundary policy: ${model.manifest.boundary_policy}. Model A shares the vocabulary but trains only on its own training text.`,"field-help"));
  const sources=el("details");sources.append(el("summary","Dataset provenance"));
  sources.append(table(["Source","Pinned revision","License"],model.manifest.sources.map(s=>[s.source_id,s.revision,s.license])));section.append(sources);
 }
 const evidence=el("details");evidence.append(el("summary","Source files and SHA-256 evidence"));
 evidence.append(table(["Project-relative source","SHA-256","Saved hash match"],model.evidence.map(e=>[e.path,e.sha256,e.matches_saved_hash===null?"Identity fields checked / current log":e.matches_saved_hash?"Yes":"No"])));section.append(evidence);
 return section;
}
function metrics(root,models){
 const section=el("section",null,"analytics-section");section.append(el("h2","Held-out results & measured cost"));
 section.append(el("p","Primary packed-test protocol: nonoverlapping context windows score each eligible shifted target once. Context resets per window, and windows may span stories separated by EOS. The first stream token and incomplete final tail are not scored."));
 const names=models.map(label),sources=[...new Set(models.flatMap(m=>Object.keys(m.test_sources)))];
 const rows=[];
 for(const source of sources){
  rows.push([`${source} · test loss (nats/token)`,...models.map(m=>m.test_sources[source]?.loss)]);
  rows.push([`${source} · character-token perplexity`,...models.map(m=>m.test_sources[source]?.perplexity)]);
  rows.push([`${source} · scored test targets`,...models.map(m=>m.test_sources[source]?.target_tokens)]);
 }
 rows.push(["Actual training target tokens",...models.map(m=>m.training_tokens)],
 ["Training session wall time (seconds)",...models.map(m=>m.training_seconds)],
 ["Generation throughput (new tokens/second)",...models.map(m=>m.generation_tokens_per_second)]);
 for(const source of [...new Set(models.flatMap(m=>Object.keys(m.exposure)))]) rows.push([`Training exposure · ${source} targets`,...models.map(m=>m.exposure[source])]);
 section.append(table(["Measurement",...names],rows));
 section.append(el("p","Perplexity = exp(mean cross-entropy in natural-log units). These are character-token metrics, including scored EOS and any UNK targets. Do not compare directly with unrelated subword-token perplexities.","field-help"));
 section.append(el("p",models[0].benchmark_definition??"Generation benchmark definition unavailable.","field-help"));
 section.append(el("p",`Benchmark environment: ${models[0].evaluation_environment.device} · ${models[0].evaluation_environment.machine} · PyTorch ${models[0].evaluation_environment.torch}. These saved repeated-trial measurements exclude model loading, unlike total interactive wait time.`,"field-help"));
 section.append(el("p",models[0].timing_definition??"Training timing definition unavailable.","field-help"));
 root.append(section);
}
function disclosure(title, content) {
 const details = el("details", null, "insight-disclosure");
 details.append(el("summary", title), content);
 return details;
}

function datasetName(source) {
 return {tinystories: "TinyStories", simplestories: "SimpleStories"}[source] ?? source;
}

function modelName(model) {
 return model.kind === "baseline" ? "TinyStories" : "TinyStories + SimpleStories";
}

function renderCombinedRun(root, model) {
 warnings(root, model.warnings);
 const summary = el("section", null, "insight-section");
 summary.append(el("h2", "TinyStories + SimpleStories"));
 summary.append(el("p", "Both datasets supplied training text for this model."));
 const card = el("div", null, "insight-card");
 card.append(table(["Training measure", "Saved value"], [
   ["Selected checkpoint update", model.checkpoint_update],
   ["Character predictions at that checkpoint", model.training_tokens],
   ["Validation loss (nats per character token)", model.validation_loss],
   ["Trainable parameters", model.parameter_count],
 ]));
 summary.append(card, el("p", "Validation measures how well the model predicts characters on separate examples. Read the continuations below to see how it handled the actual prompts.", "field-help"));
 root.append(summary);

 const samples = el("section", null, "insight-section");
 samples.append(el("h2", "What the model wrote"));
 for (const sample of model.diagnostics) {
   const card = el("article", null, "insight-card");
   card.append(el("h3", "Story opening"), el("p", sample.prompt),
     el("h3", "Generated continuation"), el("p", sample.continuation || "The model ended the story immediately."));
   samples.append(card);
 }
 root.append(samples);
 const chart = (title, points) => lossChart(title, [{label: "TinyStories + SimpleStories", points}]);
 root.append(chart("Validation loss", model.validation),
   disclosure("Show training loss", chart("Training loss", model.training)));
}

function renderExperiment(root, data, models) {
 if (!models.length) {
   root.append(el("p", "Restore the saved experiment files and reload to see results.", "inline-note"));
   return;
 }

 const results = el("section", null, "insight-section");
 results.append(el("h2", "How the models scored"));
 results.append(el("p", "Test loss measures how well a model predicts the next character in stories it did not train on. Lower is better for each dataset."));
 const groups = data.compatible ? [models] : models.map(model => [model]);
 const cards = el("div", null, "insight-cards");
 for (const group of groups) {
   const sources = new Set(group.flatMap(model => Object.keys(model.test_sources)));
   for (const source of sources) {
     const card = el("article", null, "insight-card");
     card.append(el("h3", datasetName(source)));
     const scores = el("dl", null, "insight-scores");
     for (const model of group) {
       const score = model.test_sources[source];
       const item = el("div");
       item.append(el("dt", modelName(model)), el("dd", fmt(score?.loss)));
       scores.append(item);
     }
     card.append(scores, el("p", "Test loss · nats per character token", "field-help"));
     cards.append(card);
   }
 }
 if (cards.childElementCount) results.append(cards);
 else results.append(el("p", "Held-out test scores are unavailable.", "inline-note"));
 if (data.compatible) {
   const baseline = models.find(model => model.kind === "baseline");
   const combined = models.find(model => model.kind === "combined");
   const findings = ["tinystories", "simplestories"].flatMap(source => {
     const a = baseline?.test_sources[source]?.loss, b = combined?.test_sources[source]?.loss;
     if (!Number.isFinite(a) || !Number.isFinite(b)) return [];
     return [a === b ? `The runs tied on ${datasetName(source)}.`
       : `${modelName(a < b ? baseline : combined)} scored lower on ${datasetName(source)}.`];
   });
   if (findings.length) results.append(el("p", findings.join(" "), "insight-finding"));
   results.append(el("p", "The saved comparison used one training seed. These scores measure character prediction. Story quality needs a separate check.", "field-help"));
 }
 root.append(results);

 const learning = el("section", null, "insight-section");
 learning.append(el("h2", "How the models learned"), el("p", "Training loss follows the examples used to adjust the model. Validation loss checks how well those lessons carry over to separate stories."));
 for (const group of groups) {
   if (group.length === 1) learning.append(el("h3", modelName(group[0])));
   const chart = (title, key) => lossChart(title, group.map(model => ({label: modelName(model), points: model[key]})));
   learning.append(chart("Validation loss", "validation"),
     disclosure("Show training loss", chart("Training loss", "training")));
 }
 root.append(learning);

 const details = el("section", null, "insight-section");
 details.append(el("h2", "Experiment details"));
 const measurements = el("div");
 for (const group of groups) metrics(measurements, group);
 details.append(disclosure("Test metrics, training time and throughput", measurements));
 for (const model of data.models) {
   details.append(disclosure(`${modelName(model)} · configuration and sources`, overview(model)));
 }
 root.append(details);
}

function storyArchitecture(root, model) {
 const flow = el("ol", null, "story-architecture-flow");
 for (const [title, copy] of [
   ["Read your opening", "The app breaks your sentence into small pieces of text called tokens. A token can be a word, part of a word, or punctuation."],
   ["Work out what comes next", `The text passes through ${model.architecture.layers} layers. Each layer helps the model connect details already in the story, such as who is there and what just happened.`],
   ["Write the continuation", "The model scores possible next tokens and picks one. It adds that piece to the story and repeats until it reaches an ending or the length limit."],
 ]) {
   const stage = el("li");
   stage.append(el("h2", title), el("p", copy));
   flow.append(stage);
 }
 root.append(flow);

 const training = el("section", null, "story-architecture-training");
 training.append(el("h2", "How it learned to continue a story"),
   el("p", `${model.base_model} supplied the starting weights. During local training, the model read openings from TinyStories and SimpleStories and practiced predicting the text that followed.`),
   el("p", "Training adjusted small parts of the attention layers. Attention helps the model decide which earlier details to use as it writes each new piece of text."));
 root.append(training);

 const example = model.diagnostics?.find(row => row.prompt.includes("blue ribbon") && row.continuation);
 if (example) {
   const panel = el("section", null, "story-architecture-example");
   panel.append(el("h2", "See it in a story"), el("h3", "Opening"), el("p", example.prompt, "story-prose"),
     el("h3", "From the model’s continuation"),
     el("blockquote", example.continuation.split(/(?<=[.!?])\s+/).slice(0, 2).join(" "), "story-prose"));
   root.append(panel);
 }

 const technical = el("section", null, "story-architecture-training story-architecture-technical");
 technical.append(el("h2", "How your sentence becomes a story"),
   el("p", "Here’s what happens when you press Generate, step by step."));
 const stages = el("ol", null, "architecture-flow");
 for (const [title, copy] of [
   ["Split text into tokens (tokenization)", "The app splits your opening into small pieces called tokens. A token can be a word, part of a word, or punctuation. Each token gets an ID, which the model turns into a list of numbers called an embedding."],
   ["Track word order (position information)", "The model keeps track of where tokens appear. This helps it tell the difference between ‘The dog chased the cat’ and ‘The cat chased the dog’."],
   ["Process the text (decoder layers)", "The text passes through processing stages called decoder layers. Each layer builds on the previous layer’s work to help predict what could come next."],
   ["Connect earlier details (causal attention)", "Attention helps the model use relevant details from the text so far, such as the bird and ribbon in an opening. Causal means it can look at earlier tokens, but cannot peek at future ones."],
   ["Choose the next token (sampling)", "The model scores possible next pieces. Greedy mode picks the highest-scoring one. Sampling chooses from likely options, which can make each story different."],
   ["Build the story (autoregressive generation)", "Autoregressive means writing one token at a time, using the text already written to predict the next token. The process repeats until the model chooses an end token or reaches the length limit."],
 ]) {
   const stage = el("li");
   stage.append(el("h3", title), el("p", copy));
   stages.append(stage);
 }
 technical.append(stages);
 root.append(technical);
}

function storyModelDetails(root, model, architectureOnly) {
 const a = model.architecture;
 const section = el("section", null, "insight-section");
 section.append(el("h2", architectureOnly ? "How story continuation works" : "Story continuation model"),
   el("p", `Base model: ${model.base_model}. Locally fine-tuned on TinyStories and SimpleStories.`));
 if (architectureOnly) {
   section.append(el("p", `The ${a.layers}-layer decoder reads the opening and predicts the next piece of text. It uses a subword vocabulary, so a token can be a whole word or part of one.`),
     el("p", "Training adjusts small attention adapters using the parts of stories that follow their openings."));
 } else {
   section.append(table(["Training measure", "Saved value"], [
     ["Selected update", model.checkpoint_update],
     ["Trainable adapter parameters", model.trainable_parameters],
     ["Validation loss before fine-tuning", model.validation_before],
     ["Validation loss after fine-tuning", model.validation_loss],
   ]));
   section.append(storyCharts(model));
   for (const sample of model.diagnostics.slice(0, 4)) {
     const card = el("article", null, "insight-card");
     card.append(el("h3", "Opening"), el("p", sample.prompt),
       el("h3", "Actual continuation"), el("p", sample.continuation, "story-prose"));
     section.append(card);
   }
 }
 root.append(section);
}

function architecture(root,model) {
 const a=model.architecture;root.append(el("h2",label(model)));
 const flow=el("ol",null,"architecture-flow");
 const stages=[
 ["Token IDs","[batch, sequence] · integer character and special-token IDs."],
 ["Token + learned positional embeddings",`Token lookup plus positions 0…sequence−1 → [batch, sequence, ${a.embedding_dim}]. Embedding dropout ${a.dropout} during training.`],
 [`${a.num_layers} causal decoder blocks`,`Pre-normalization · ${a.num_heads} heads · ${a.embedding_dim/a.num_heads} features per head.`],
 ["Final custom layer normalization",`Affine scale and bias; epsilon ${a.layer_norm_epsilon}.`],
 ["Vocabulary scores",`Bias-free linear projection → [batch, sequence, ${a.vocab_size}]. ${a.tie_embeddings?"Output weights tied to token embeddings.":"Separate output weights."}`],
 ["Next-token selection","Inference uses the last position's scores: greedy argmax or temperature/top-k sampling, then appends the selected ID. EOS can stop generation."]
 ];
 for(const [title,description] of stages){const li=el("li");li.append(el("h3",title),el("p",description));flow.append(li);}root.append(flow);

 const explanation = el("section", null, "analytics-section");
 explanation.append(el("h2", "How the model continues your story"));
 const steps = el("ol", null, "architecture-flow");
 for (const [title, description] of [
   ["Turn characters into numbers", "The model looks up a number for each character in your opening. For example, the letters in ‘cat’ each have their own number."],
   ["Keep track of the order", "The model gives each character a set of learned features and adds information about its position. This helps it tell ‘cat’ from ‘act’."],
   ["Look back at what has been written", "Each decoder block uses the current character and the ones before it to build a better picture of the text so far. It has several attention heads that can focus on different patterns."],
   ["Put the features on a consistent scale", "The model adjusts the numbers from the last decoder block so that unusually large or small values are less likely to dominate the next step."],
   ["Score the possible next characters", "The model gives each possible next character a score. If the text ends in ‘nes’, for example, it might give ‘t’ a high score to complete ‘nest’."],
   ["Pick a character and repeat", "Greedy mode picks the highest-scoring choice. Sampling uses the scores to make a weighted random choice. The model adds the chosen character and repeats until it chooses the end-of-story marker or reaches the length limit."],
 ]) {
   const step = el("li");
   step.append(el("h3", title), el("p", description));
   steps.append(step);
 }
 explanation.append(steps);
 root.append(explanation);

}
function mountInsights(root,{architectureOnly=false,architectureModelId,fetcher=fetch}={}) {
 let disposed=false;
 root.replaceChildren(el("h1",architectureOnly?"How Transformer writes a story":"Model insights"),
 el("p",architectureOnly?"Follow your opening from the first sentence to the generated continuation.":"Explore training results and story continuations.","page-description"));
 const body=el("div",null,architectureOnly?"story-architecture":null);root.append(body);
 async function load(){
  if(retry.disabled)return;retry.disabled=true;
  body.setAttribute("aria-busy","true");
  const loading=el("p","Reading registered experiment artifacts…","inline-note");loading.setAttribute("role","status");body.replaceChildren(loading);
  try{
   const response=await fetcher("/api/v1/analytics",{credentials:"omit"});
   if(!response.ok)throw Error("Unavailable");const data=await response.json();if(disposed)return;
   if(!Array.isArray(data.models)||!Array.isArray(data.warnings))throw Error("Malformed response");
   body.replaceChildren();warnings(body,data.warnings);
   const valid=data.models.filter(m=>m.status==="available");
   if(architectureOnly){
    const model = valid.find(entry => entry.model_id === architectureModelId);
    if (model?.scope === "story_continuation") storyArchitecture(body, model);
    else if (model) architecture(body, model);
    else body.append(el("p", "The generation model configuration is unavailable. Restore its registered artifacts and reload."));
    return;
   }
   const story = valid.find(model => model.scope === "story_continuation");
   if (story) { storyModelDetails(body, story, false); return; }
   const combined = valid.find(model => model.scope === "validation_and_diagnostic_samples");
   if (combined) renderCombinedRun(body, combined);
   else renderExperiment(body, data, valid);
  }catch{if(!disposed)body.replaceChildren(el("p","Analytics unavailable. Check the local service and restore the registered experiment artifacts, then retry.","inline-note"));}
  finally { retry.disabled=false;body.setAttribute("aria-busy","false"); }
 }
 const retry=el("button","Reload artifact evidence","button button-secondary");retry.type="button";retry.onclick=load;if(!architectureOnly)root.append(retry);load();
 return()=>{disposed=true;};
}
return { mountInsights };
})();

// Session History
const session_history = (() => {
function renderHistory(container, state, onRestore) {
  container.replaceChildren();
  if (!state.history.length) {
    const empty = document.createElement("p");
    empty.className = "history-empty";
    empty.textContent = "No saved items yet. Save an opening to return to it during this session.";
    container.append(empty);
    return;
  }
  const list = document.createElement("ol");
  list.className = "history-list";
  for (const entry of state.history) {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "history-item";
    button.disabled = state.outcome === "loading";
    const label = document.createElement("span");
    label.className = "history-kind";
    label.textContent = entry.kind === "opening" ? "Saved opening" : `Continuation · ${entry.result.modelLabel}`;
    const text = document.createElement("span");
    text.className = "history-title";
    const opening = Array.from(entry.draft.prompt.trim());
    text.textContent = opening.slice(0, 90).join("") + (opening.length > 90 ? "…" : "");
    const time = document.createElement("time");
    time.dateTime = new Date(entry.savedAt).toISOString();
    time.textContent = new Date(entry.savedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    button.append(label, text, time);
    button.addEventListener("click", () => onRestore(entry.id));
    item.append(button);
    list.append(item);
  }
  container.append(list);
}
return { renderHistory };
})();

// Generation Metadata
const generation_metadata = (() => {
function renderGenerationMetadata(metadata) {
  const details = document.createElement("details");
  details.className = "generation-metadata";
  const summary = document.createElement("summary");
  summary.textContent = "Generation details";
  const list = document.createElement("dl");
  const settings = metadata.sampling_settings;
  const rows = [
    ["Generated tokens", metadata.generated_token_count],
    ["Generation time", `${metadata.generation_duration_seconds.toFixed(2)} s`],
    ["Generation mode", settings.mode === "greedy" ? "Greedy" : "Sampling"],
  ];
  if (settings.mode === "sample") rows.push(["Temperature", settings.temperature]);
  for (const [label, value] of rows) {
    const row = document.createElement("div");
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = label;
    description.textContent = String(value);
    row.append(term, description);
    list.append(row);
  }
  details.append(summary, list);
  return details;
}
return { renderGenerationMetadata };
})();

// Session State
const session_state = (() => {
// State is private to this browser tab; transport lives in generation_client.js.
const PROMPT_LIMIT = 16384;
const HISTORY_LIMIT = 10;
const EXAMPLE_PROMPTS = Object.freeze([
  { label: "The blue ribbon", text: "The little bird found a blue ribbon beside the old oak tree." },
  { label: "A new friend", text: "Mia saw that her new friend had no one to play with." },
  { label: "The little box", text: "Tom opened a small box and heard a quiet tapping inside." },
]);

function validatePrompt(prompt) {
  if (typeof prompt !== "string") return "Enter a story opening as text.";
  if (!prompt.trim()) return "Write an opening or choose an example to begin.";
  if (Array.from(prompt).length > PROMPT_LIMIT) return "Keep your opening within 16,384 characters. Shorten it and try again.";
  if (Array.from(prompt).some(char => char.codePointAt(0) >= 0xd800 && char.codePointAt(0) <= 0xdfff)) {
    return "This opening contains an invalid text character. Remove it or paste plain text.";
  }
  return "";
}

function deriveView(state) {
  if (state.outcome === "loading") return "loading";
  if ((state.validationVisible || Array.from(state.draft.prompt).length > PROMPT_LIMIT) && validatePrompt(state.draft.prompt)) return "validation_error";
  if (["request_error", "missing_checkpoint", "backend_unavailable"].includes(state.outcome)) return state.outcome;
  if (state.connection === "unreachable") return "backend_unavailable";
  if (state.result) return "success";
  return validatePrompt(state.draft.prompt) ? "empty" : "ready";
}

function createSession({ now = () => Date.now(), fixedModel = "" } = {}) {
  let state = {
    draft: { prompt: "", model: fixedModel, mode: fixedModel ? "sample" : "greedy", variation: fixedModel ? "0.65" : "0.8", length: fixedModel.startsWith("transformer-story-continuation-") ? "250" : fixedModel ? "384" : "128", topK: fixedModel.startsWith("transformer-story-continuation-") ? "20" : fixedModel ? "10" : "", seed: "" },
    connection: "checking", outcome: null, result: null, validationVisible: false,
    history: [], undoDraft: null, models: [], registryFingerprint: null, error: null, serviceBusy: false,
  };
  let sequence = 0;
  const listeners = new Set();
  const snapshot = () => structuredClone(state);
  const notify = () => listeners.forEach(listener => listener(snapshot()));
  function editable() {
    if (state.outcome === "loading") throw new Error("Wait for the current operation to finish.");
  }
  function checkedDraft(draft) {
    if (typeof draft.prompt !== "string" || typeof draft.model !== "string" || (draft.model !== "" && !/^[a-z0-9][a-z0-9_-]{0,79}$/.test(draft.model))
      || !["greedy", "sample"].includes(draft.mode) || typeof draft.length !== "string"
      || typeof draft.variation !== "string" || typeof draft.topK !== "string" || typeof draft.seed !== "string") throw new TypeError("Invalid draft settings.");
    return { ...draft, model: fixedModel || draft.model };
  }
  function addHistory(entry) {
    state.history = [{ ...entry, id: `entry-${++sequence}`, savedAt: now() }, ...state.history].slice(0, HISTORY_LIMIT);
  }
  return {
    snapshot,
    subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); },
    updateDraft(patch) {
      editable();
      state.draft = checkedDraft({ ...state.draft, ...patch });
      state.undoDraft = null;
      if (state.error) { state.error = null; state.outcome = null; }
      notify();
    },
    replaceDraft(patch) {
      editable();
      const next = checkedDraft({ ...state.draft, ...patch });
      state.undoDraft = { ...state.draft };
      state.draft = next;
      state.validationVisible = false;
      state.error = null;
      state.outcome = null;
      notify();
    },
    undoReplacement() {
      editable();
      if (state.undoDraft) { state.draft = state.undoDraft; state.undoDraft = null; state.validationVisible = false; state.error = null; state.outcome = null; notify(); }
    },
    validate() { state.validationVisible = true; notify(); return !validatePrompt(state.draft.prompt); },
    setConnection(value) {
      if (!["checking", "not_connected", "unreachable", "ready"].includes(value)) throw new TypeError("Unknown interface connection state.");
      state.connection = value;
      notify();
    },
    setCatalog(catalog, busy = false) {
      state.models = structuredClone(catalog.models);
      state.registryFingerprint = catalog.registry_fingerprint;
      state.serviceBusy = busy;
      state.connection = "ready";
      if (!state.draft.model) {
        const preferred = state.models.find(model => model.kind === "english" && model.availability === "registered");
        state.draft.model = preferred?.model_id
          ?? state.models.find(model => model.availability === "registered")?.model_id
          ?? state.models[0]?.model_id ?? "";
        if (preferred) {
          Object.assign(state.draft, { mode: "sample", variation: "0.65", topK: "10" });
        }
      }
      if (["service_unavailable", "artifact_unavailable", "model_not_found"].includes(state.error?.code)) { state.error = null; state.outcome = null; }
      if (state.draft.model && !state.models.some(model => model.model_id === state.draft.model)) {
        state.error = { code: "model_not_found", message: "Your selected model is no longer registered. Restore its registry entry and check the service again." };
        state.outcome = "request_error";
      }
      notify();
    },
    setFailure(error) {
      state.error = structuredClone(error);
      state.outcome = error.code === "artifact_unavailable" ? "missing_checkpoint"
        : error.code === "service_unavailable" ? "backend_unavailable" : "request_error";
      if (error.code === "artifact_unavailable") {
        const model = state.models.find(item => item.model_id === state.draft.model);
        if (model) { model.availability = "unavailable"; model.unavailable_reason = error.message; }
      }
      notify();
    },
    clearHistory() { editable(); state.history = []; notify(); },
    recordGeneration(draft, result) { addHistory({ kind: "continuation", draft: structuredClone(draft), result: structuredClone(result) }); notify(); },
    saveOpening() {
      editable();
      if (validatePrompt(state.draft.prompt)) throw new Error(validatePrompt(state.draft.prompt));
      const existing = state.history.find(item => item.kind === "opening" && JSON.stringify(item.draft) === JSON.stringify(state.draft));
      if (existing) return false;
      addHistory({ kind: "opening", draft: { ...state.draft } });
      notify();
      return true;
    },
    restoreHistory(id) {
      editable();
      const item = state.history.find(entry => entry.id === id);
      if (!item) throw new Error("This item is no longer in the current tab's history.");
      state.undoDraft = { ...state.draft };
      state.draft = checkedDraft(item.draft);
      state.validationVisible = false;
      state.outcome = null;
      state.error = null;
      state.result = item.kind === "continuation" ? structuredClone(item.result) : null;
      notify();
    },
    setOutcome(kind, result = null) {
      if (![null, "loading", "success", "backend_unavailable", "missing_checkpoint"].includes(kind)) throw new TypeError("Unknown output state.");
      if (kind === "success") {
        if (!result || typeof result.prompt !== "string" || typeof result.continuation !== "string"
          || typeof result.modelLabel !== "string" || !result.modelLabel.trim()
          || Array.from(result.prompt).length > PROMPT_LIMIT || result.continuation.length > 32768
          || !["eos", "max_new_tokens"].includes(result.stopReason)) throw new TypeError("Invalid completed output.");
        state.result = structuredClone(result);
        addHistory({ kind: "continuation", draft: { ...state.draft, prompt: result.prompt }, result: structuredClone(result) });
      }
      state.outcome = kind;
      state.error = null;
      state.validationVisible = false;
      notify();
    },
  };
}
return { PROMPT_LIMIT, HISTORY_LIMIT, EXAMPLE_PROMPTS, validatePrompt, deriveView, createSession };
})();

// Generation Client
const generation_client = (() => {
const { PROMPT_LIMIT, validatePrompt } = session_state;
const hash = value => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
const integer = (value, min, max) => Number.isInteger(value) && value >= min && value <= max;
const seconds = value => typeof value === "number" && Number.isFinite(value) && value >= 0;
const seed = value => value === null || (typeof value === "string" && /^(0|[1-9][0-9]{0,18})$/.test(value) && BigInt(value) < 2n ** 63n);

function validateCatalog(value) {
  if (!value || !Array.isArray(value.models) || !value.models.length || !hash(value.registry_fingerprint)) throw new Error("Invalid model catalog.");
  const ids = new Set();
  for (const model of value.models) {
    if (!model || typeof model.model_id !== "string" || !/^[a-z0-9][a-z0-9_-]{0,79}$/.test(model.model_id) || ids.has(model.model_id)
      || !["baseline", "combined", "english"].includes(model.kind) || (typeof model.display_name !== "string" || !model.display_name.trim())
      || !["registered", "unavailable"].includes(model.availability)
      || !(model.unavailable_reason === null || typeof model.unavailable_reason === "string")
      || !hash(model.checkpoint?.sha256) || !integer(model.checkpoint.update, 1, Number.MAX_SAFE_INTEGER)
      || !hash(model.tokenizer_fingerprint)) throw new Error("Invalid model catalog.");
    ids.add(model.model_id);
  }
  return value;
}

function requestFromDraft(draft) {
  const error = validatePrompt(draft.prompt);
  if (error) throw new Error(error);
  const max = Number(draft.length);
  if (!/^\d+$/.test(draft.length) || !integer(max, 1, 1024) || !["greedy", "sample"].includes(draft.mode)) throw new Error("Choose valid generation settings.");
  const settings = { mode: draft.mode, max_new_tokens: max, unknown_policy: "error" };
  if (draft.mode === "sample") {
    const temperature = Number(draft.variation);
    if (!Number.isFinite(temperature) || temperature <= 0) throw new Error("Temperature must be a finite number greater than zero.");
    settings.temperature = temperature;
    if (draft.topK) {
      if (!/^\d+$/.test(draft.topK) || !integer(Number(draft.topK), 1, 2147483647)) throw new Error("Top-k must be a positive integer, or blank for all tokens.");
      settings.top_k = Number(draft.topK);
    }
    if (draft.seed) {
      if (!seed(draft.seed)) throw new Error("Seed must be an integer from 0 to 9223372036854775807.");
      settings.seed = draft.seed;
    }
  }
  return { model_id: draft.model, prompt: draft.prompt, settings };
}

function validateResult(result, request, catalog) {
  const model = catalog.models.find(row => row.model_id === request.model_id);
  const settings = result?.sampling_settings;
  if (!result || !model || result.model_id !== request.model_id || result.prompt !== request.prompt
    || result.model_kind !== model.kind || result.checkpoint?.sha256 !== model.checkpoint.sha256
    || result.checkpoint.update !== model.checkpoint.update
    || result.tokenizer_fingerprint !== model.tokenizer_fingerprint
    || result.registry_fingerprint !== catalog.registry_fingerprint
    || typeof result.continuation !== "string" || result.continuation.length > 10240
    || !integer(result.generated_token_count, 0, request.settings.max_new_tokens)
    || !["eos", "max_new_tokens"].includes(result.stop_reason)
    || !seconds(result.generation_duration_seconds) || !seconds(result.model_load_seconds)
    || !integer(result.context_length, 1, Number.MAX_SAFE_INTEGER)
    || !integer(result.initial_prompt_tokens_truncated, 0, PROMPT_LIMIT)
    || !integer(result.unknown_prompt_tokens, 0, PROMPT_LIMIT)
    || !["cpu", "mps"].includes(result.device) || typeof result.cache_hit !== "boolean"
    || typeof result.request_id !== "string" || !/^[a-f0-9]{32}$/.test(result.request_id)
    || !settings || settings.mode !== request.settings.mode || settings.max_new_tokens !== request.settings.max_new_tokens
    || settings.temperature !== (request.settings.temperature ?? 0) || settings.top_k !== (request.settings.top_k ?? null)
    || settings.seed !== (request.settings.seed ?? null) || !seed(settings.effective_seed) || settings.stop_at_eos !== true
    || (request.settings.seed != null && settings.effective_seed !== request.settings.seed)
    || settings.unknown_policy !== request.settings.unknown_policy
    || (settings.mode === "greedy" ? settings.effective_seed !== null : settings.effective_seed === null)) {
    throw new Error("The generation response did not match the requested model and settings.");
  }
  return { prompt: result.prompt, continuation: result.continuation, modelLabel: model.display_name,
    stopReason: result.stop_reason, metadata: structuredClone(result) };
}

async function readResponse(response) {
  const body = await response.json();
  if (!response.ok) {
    if (!body?.error || typeof body.error.code !== "string" || typeof body.error.message !== "string") throw new Error("Invalid server error.");
    throw Object.assign(new Error(body.error.message), { apiError: body.error });
  }
  return body;
}

function createGenerationClient(session, { fetcher = fetch, statusTimeoutMs = 4000 } = {}) {
  let refreshing = null, submitting = false;
  async function refresh() {
    if (refreshing || submitting) return refreshing;
    session.setConnection("checking");
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), statusTimeoutMs);
    refreshing = (async () => {
      try {
        const options = { signal: controller.signal, credentials: "omit" };
        const status = await readResponse(await fetcher("/api/v1/status", options));
        if (status.phase !== 15 || status.contract_version !== "2.0.0" || typeof status.busy !== "boolean") throw new Error("Incompatible local service.");
        const catalog = validateCatalog(await readResponse(await fetcher("/api/v1/models", options)));
        if (!status.generation_enabled) throw new Error("Inference is unavailable.");
        session.setCatalog(catalog, status.busy);
      } catch (error) {
        session.setConnection("unreachable");
        session.setFailure({ code: "service_unavailable", message: error.apiError?.message
          ?? "The local service is unavailable or incompatible. Start the application and check the service again. Your opening is preserved." });
      } finally { clearTimeout(timer); refreshing = null; }
    })();
    return refreshing;
  }
  async function submit() {
    if (submitting) return;
    const snapshot = session.snapshot();
    if (!session.validate()) return;
    const model = snapshot.models.find(row => row.model_id === snapshot.draft.model);
    if (snapshot.connection !== "ready" || model?.availability !== "registered" || snapshot.serviceBusy) return;
    let request;
    try { request = requestFromDraft(snapshot.draft); }
    catch (error) { session.setFailure({ code: "invalid_request", message: error.message }); return; }
    const catalog = { models: snapshot.models, registry_fingerprint: snapshot.registryFingerprint };
    submitting = true;
    session.setOutcome("loading");
    try {
      const raw = await readResponse(await fetcher("/api/v1/generations", {
        method: "POST", credentials: "omit", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request),
      }));
      session.setOutcome("success", validateResult(raw, request, catalog));
    } catch (error) {
      session.setFailure(error.apiError ?? { code: "service_unavailable", message:
        "The response could not be verified or the connection was lost. The previous request may still be running. Check the service before trying again; your opening is preserved." });
      if (!error.apiError) session.setConnection("unreachable");
    } finally { submitting = false; }
  }
  return { refresh, submit };
}
return { validateCatalog, requestFromDraft, validateResult, createGenerationClient };
})();

// Model Picker
const model_picker = (() => {
function renderModelPicker(container, models) {
  container.replaceChildren();
  if (!models.length) {
    const message = document.createElement("p");
    message.className = "field-help";
    message.textContent = "No models available yet. Check the local service.";
    container.append(message);
  }
  for (const model of models) {
    const label = document.createElement("label");
    label.className = "model-choice";
    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "model";
    radio.value = model.model_id;
    radio.dataset.available = String(model.availability === "registered");
    const description = document.createElement("span");
    const name = document.createElement("strong");
    name.textContent = model.display_name;
    const note = document.createElement("span");
    note.textContent = `${({ baseline: "Model A", combined: "Model B", english: "Extended run" })[model.kind]} · ${model.availability === "registered" ? "Registered model" : "Files unavailable"}`;
    description.append(name, note);
    if (model.unavailable_reason) label.title = model.unavailable_reason;
    label.append(radio, description);
    container.append(label);
  }
}
return { renderModelPicker };
})();

// Components
const components = (() => {
const stateKinds = new Set(["loading", "empty", "success", "error"]);

function renderState(container, { kind, title, description, action }) {
  if (!stateKinds.has(kind)) throw new TypeError("Unknown interface state.");
  const region = document.createElement("section");
  region.className = "state-message";
  region.dataset.state = kind;
  region.setAttribute("role", kind === "error" ? "alert" : "status");
  const heading = document.createElement("h3");
  heading.textContent = title;
  const body = document.createElement("p");
  body.textContent = description;
  region.append(heading, body);
  if (action) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "button button-secondary";
    button.textContent = action.label;
    button.addEventListener("click", action.onClick);
    region.append(button);
  }
  container.replaceChildren(region);
}
return { renderState };
})();

// Story Output
const story_output = (() => {
const { renderGenerationMetadata } = generation_metadata;
const { renderState } = components;
const { deriveView, validatePrompt } = session_state;
const messages = {
  empty: ["empty", "A space for what comes next.", "Write an opening above or choose an example. Generate to see what follows."],
  ready: ["empty", "Your opening is ready.", "Choose your preferences, then generate a continuation. You can also save this opening for later."],
  loading: ["loading", "Generating the continuation…", "Waiting for the complete text. Keep this tab open; there is no streaming or cancellation control."],
  backend_unavailable: ["error", "The local service is unavailable.", "Start the local application, then check the interface again. Your opening and saved session items are still here."],
  missing_checkpoint: ["error", "This model's checkpoint is unavailable.", "Restore the registered checkpoint and check the service again. Your opening is preserved."],
};

function renderStoryOutput(container, state, { onRetry } = {}) {
  const view = deriveView(state);
  container.replaceChildren();
  container.dataset.view = view;
  const feedback = document.createElement("div");
  container.append(feedback);
  if (state.error && view !== "loading") {
    renderState(feedback, { kind: "error", title: view === "missing_checkpoint" ? "This model is unavailable." : "Could not generate yet.", description: state.error.message,
      ...(view === "backend_unavailable" && onRetry ? { action: { label: "Check service", onClick: onRetry } } : {}),
    });
  } else if (view === "validation_error") {
    renderState(feedback, { kind: "error", title: "Check your opening.", description: validatePrompt(state.draft.prompt) });
  } else if (view !== "success") {
    const [kind, title, description] = messages[view];
    renderState(feedback, { kind, title, description,
      ...(view === "backend_unavailable" && onRetry ? { action: { label: "Check interface", onClick: onRetry } } : {}),
    });
  }
  if (!state.result) return;
  const result = state.result;
  const article = document.createElement("article");
  article.className = "story-result";
  article.setAttribute("aria-label", view === "success" ? "Completed continuation" : "Previous continuation");
  const identity = document.createElement("p");
  identity.className = "result-identity";
  identity.textContent = `${view === "success" ? "Completed" : "Previous result"} · ${result.modelLabel}`;
  const opening = document.createElement("details");
  opening.className = "result-opening";
  const summary = document.createElement("summary");
  summary.textContent = "Original prompt";
  const prompt = document.createElement("p");
  prompt.className = "story-prose prompt-prose";
  prompt.textContent = result.prompt;
  opening.append(summary, prompt);
  const headingRow = document.createElement("div");
  headingRow.className = "section-heading";
  const heading = document.createElement("h3");
  heading.textContent = "Continuation";
  const copy = document.createElement("button");
  copy.className = "button button-secondary";
  copy.type = "button";
  copy.textContent = "Copy continuation";
  copy.disabled = !result.continuation;
  headingRow.append(heading, copy);
  const prose = document.createElement("div");
  prose.className = "story-prose continuation-prose";
  prose.tabIndex = 0;
  prose.setAttribute("aria-label", "Continuation text");
  prose.textContent = result.continuation || (result.stopReason === "eos" ? "The model ended the story before adding visible text." : "No new tokens were requested.");
  const copyFeedback = document.createElement("p");
  copyFeedback.className = "copy-feedback";
  copyFeedback.setAttribute("role", "status");
  copy.addEventListener("click", async () => {
    copy.disabled = true;
    try {
      await navigator.clipboard.writeText(result.continuation);
      copyFeedback.textContent = "Continuation copied.";
    } catch {
      copyFeedback.textContent = "Copy was unavailable. Select the continuation text and copy it manually.";
      prose.focus();
    } finally { copy.disabled = false; }
  });
  const ending = document.createElement("p");
  ending.className = "result-ending";
  ending.textContent = result.stopReason === "eos" ? "The model finished the story before the maximum token limit." : "Length limit reached. The story may be unfinished.";
  article.append(identity, opening, headingRow, prose, copyFeedback, ending);
  if (result.metadata) {
    if (result.metadata.initial_prompt_tokens_truncated) {
      const note = document.createElement("p");
      note.className = "inline-note";
      note.textContent = `This opening exceeded the model's context. Generation began with its last ${result.metadata.context_length} character tokens.`;
      article.append(note);
    }
    article.append(renderGenerationMetadata(result.metadata));
  }
  container.append(article);
}
return { renderStoryOutput };
})();

// Generate Page
const generate_page = (() => {
const { EXAMPLE_PROMPTS, deriveView, validatePrompt, PROMPT_LIMIT } = session_state;
const { renderStoryOutput } = story_output;
const { renderHistory } = session_history;
function mountGeneratePage(root, session, { onRetry, onGenerate, actionLabel = "Generate continuation", onRestore } = {}) {
  const find = selector => root.querySelector(selector);
  const prompt = find("#prompt");
  const flash = find("#draft-feedback");
  const mode = find("#writing-mode");
  const variation = find("#variation");
  const length = find("#story-length");
  let outputKey = "", historyKey = "";
  function feedback(message) { flash.textContent = message; }
  function showRestoredOpening(message) {
    feedback(message);
    prompt.focus();
    prompt.scrollIntoView({ block: "nearest" });
  }
  const examples = find("#example-prompts");
  for (const example of EXAMPLE_PROMPTS) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "button example-prompt";
    button.textContent = example.label;
    button.title = example.text;
    button.addEventListener("click", () => {
      session.replaceDraft({ prompt: example.text });
      showRestoredOpening("Example opening added. Use Undo to restore your previous draft.");
    });
    examples.append(button);
  }
  prompt.addEventListener("input", () => {
    session.updateDraft({ prompt: prompt.value });
    feedback("");
  });
  prompt.addEventListener("blur", () => session.validate());
  find("#clear-draft").addEventListener("click", () => {
    session.replaceDraft({ prompt: "" });
    showRestoredOpening("Opening cleared. Use Undo to restore it.");
  });
  find("#undo-draft").addEventListener("click", () => {
    session.undoReplacement();
    showRestoredOpening("Previous opening restored.");
  });
  mode.addEventListener("change", () => session.updateDraft({ mode: mode.value }));
  variation.addEventListener("input", () => session.updateDraft({ variation: variation.value }));
  length.addEventListener("input", () => session.updateDraft({ length: length.value }));
  find("#generation-form").addEventListener("submit", event => {
    event.preventDefault();
    if (!session.validate()) { prompt.focus(); return; }
    onGenerate?.();
  });
  find("#clear-history").addEventListener("click", () => { session.clearHistory(); feedback("Session history cleared."); });
  find("#check-interface").addEventListener("click", () => onRetry?.());

  function update(state) {
    const busy = state.outcome === "loading";
    const selected = state.models.find(model => model.model_id === state.draft.model);
    if (prompt.value !== state.draft.prompt) prompt.value = state.draft.prompt;
    const count = Array.from(state.draft.prompt).length;
    find("#prompt-count").textContent = `${count.toLocaleString()} / ${PROMPT_LIMIT.toLocaleString()} characters`;
    const localError = validatePrompt(state.draft.prompt);
    const error = localError || (state.error?.field === "prompt" ? state.error.message : "");
    const showError = Boolean(error && (state.validationVisible || count > PROMPT_LIMIT || state.error?.field === "prompt"));
    prompt.setAttribute("aria-invalid", String(showError));
    find("#prompt-error").textContent = showError ? error : "";
    find("#prompt-error").hidden = !showError;
    for (const control of root.querySelectorAll("#prompt, input, select, #example-prompts button")) control.disabled = busy;
    find("#clear-draft").disabled = busy || !count;
    find("#undo-draft").hidden = !state.undoDraft;
    find("#undo-draft").disabled = busy;
    mode.value = state.draft.mode;
    variation.value = state.draft.variation;
    length.value = state.draft.length;
    find("#clear-history").disabled = busy || !state.history.length;
    find("#variation-field").hidden = state.draft.mode !== "sample";
    find("#variation-value").textContent = Number(state.draft.variation).toFixed(1);
    find("#style-help").textContent = state.draft.mode === "greedy"
      ? "Chooses the most likely next token each time."
      : "Samples from possible next tokens for more varied continuations.";
    find("#generate-action").textContent = busy ? "Generating…" : actionLabel;
    find("#generate-action").disabled = busy || Boolean(localError) || !onGenerate || state.connection !== "ready" || state.serviceBusy || selected?.availability !== "registered";
    const serviceReady = state.connection === "ready" && !state.serviceBusy && selected?.availability === "registered";
    const retryNeeded = !busy && !serviceReady && state.connection !== "checking";
    find("#service-status").textContent = busy ? "Writing your continuation…" : retryNeeded
      ? (state.serviceBusy ? "Another generation is running." : "The model is unavailable.")
      : "";
    find("#check-interface").hidden = !retryNeeded;
    find("#check-interface").disabled = !onRetry;
    find("#generation-note").textContent = busy || serviceReady ? ""
      : selected?.unavailable_reason ?? (state.connection === "checking" ? "Connecting to the local model…" : "");
    const view = deriveView(state);
    find(".continuation-area").hidden = !state.result && ["empty", "ready"].includes(view);
    find("#output-state-label").textContent = {
      empty: "Not started", ready: "Opening ready", loading: "In progress", success: "Complete",
      validation_error: "Check your opening", request_error: "Could not generate", backend_unavailable: "Unavailable", missing_checkpoint: "Model unavailable",
    }[view];
    const nextOutputKey = JSON.stringify([view, state.result, state.error, state.connection, showError ? error : ""]);
    if (nextOutputKey !== outputKey) {
      renderStoryOutput(find("#story-output"), state, { onRetry });
      outputKey = nextOutputKey;
    }
    find("#story-output").setAttribute("aria-busy", String(busy));
    const nextHistoryKey = JSON.stringify([state.history, busy]);
    if (nextHistoryKey !== historyKey) {
      renderHistory(find("#session-items"), state, id => {
        session.restoreHistory(id);
        onRestore?.();
        showRestoredOpening("Session item restored. Use Undo to restore your previous opening.");
      });
      historyKey = nextHistoryKey;
    }
    find("#composer-history").hidden = !state.history.length;
    find("#history-count").textContent = `${state.history.length} / 10`;
  }
  update(session.snapshot());
  return session.subscribe(update);
}
return { mountGeneratePage };
})();

// Comparison Page
const comparison_page = (() => {
const { createSession, validatePrompt } = session_state;
const { createGenerationClient, requestFromDraft } = generation_client;
const { mountGeneratePage } = generate_page;
const { renderStoryOutput } = story_output;
const { renderState } = components;
function createComparison(session, { fetcher = fetch } = {}) {
  const comparison = { results: [], running: false, snapshot: null };
  comparison.run = async () => {
    if (comparison.running || session.snapshot().outcome === "loading") return;
    const state = session.snapshot();
    if (!session.validate()) return;
    try { requestFromDraft(state.draft); }
    catch (error) { session.setFailure({ code: "invalid_request", message: error.message }); return; }
    const pair = ["baseline", "combined"].map(kind => state.models.find(model => model.kind === kind));
    if (pair.some(model => !model || model.availability !== "registered")) {
      session.setFailure({ code: "model_not_found", message: "Comparison needs both registered models. Check the service and restore missing model files before retrying." }); return;
    }
    if (state.connection !== "ready") return;
    const draft = structuredClone(state.draft);
    // Ek seed dono requests mein same rahe; independent random seeds comparison ko badal denge.
    if (draft.mode === "sample" && !draft.seed) {
      const words = crypto.getRandomValues(new Uint32Array(2));
      draft.seed = ((BigInt(words[0] & 0x7fffffff) << 32n) | BigInt(words[1])).toString();
    }
    comparison.snapshot = structuredClone(draft);
    comparison.results = [];
    comparison.running = true;
    session.setOutcome("loading");
    try {
      for (const model of pair) {
        const child = createSession();
        child.setCatalog({ models: state.models, registry_fingerprint: state.registryFingerprint });
        child.updateDraft({ ...draft, model: model.model_id });
        await createGenerationClient(child, { fetcher }).submit();
        const outcome = child.snapshot();
        comparison.results.push({ model, state: outcome });
        if (outcome.outcome === "success") session.recordGeneration(outcome.draft, outcome.result);
        // A lost response may still own the accelerator: never automatically start B then.
        if (outcome.error?.code === "service_unavailable" || outcome.error?.code === "busy") break;
      }
    } finally { comparison.running = false; session.setOutcome(null); }
  };
  return comparison;
}

function mountComparison(root, session, comparison, onRetry) {
  root.querySelector("h1").textContent = "One opening. Two possibilities.";
  root.querySelector(".page-description").textContent = "Both models run sequentially with the same opening and settings. A single random sample is not a definitive quality test. With sampling and a blank seed, one seed is chosen and shared by both requests.";
  root.querySelector(".model-picker").hidden = true;
  const panels = document.createElement("div");
  panels.className = "comparison-grid comparison-results";
  const feedback = document.createElement("div");
  feedback.setAttribute("role", "status");
  root.append(feedback, panels);
  const stopEditor = mountGeneratePage(root, session, { onRetry, onGenerate: comparison.run, actionLabel: "Compare both models", onRestore: () => { location.hash = "/generate"; } });
  root.querySelector(".continuation-area").hidden = true;
  let key = "";
  function update() {
    const state = session.snapshot();
    const complete = ["baseline", "combined"].every(kind => state.models.some(m => m.kind === kind && m.availability === "registered"));
    root.querySelector("#generate-action").disabled = comparison.running || state.outcome === "loading" || state.connection !== "ready" || state.serviceBusy || !complete || Boolean(validatePrompt(state.draft.prompt));
    feedback.textContent = state.error?.message ?? (!complete ? "Comparison requires both Model A and Model B. Restore missing checkpoints and check the service again." : "");
    const next = JSON.stringify([comparison.results, comparison.running]);
    if (next === key) return;
    key = next; panels.replaceChildren();
    for (const [index, name] of ["Model A · TinyStories", "Model B · TinyStories + SimpleStories"].entries()) {
      const section = document.createElement("section");
      const heading = document.createElement("h2"); heading.textContent = name; section.append(heading);
      const body = document.createElement("div"); section.append(body); panels.append(section);
      const entry = comparison.results[index];
      if (entry) {
        renderStoryOutput(body, entry.state);
        if (entry.state.result?.metadata) {
          const m = entry.state.result.metadata, summary = document.createElement("p");
          summary.className = "field-help";
          summary.textContent = `${m.model_id} · ${m.generated_token_count} tokens · ${m.generation_duration_seconds.toFixed(3)} s generation (loading excluded) · ${m.stop_reason}`;
          section.insertBefore(summary, body);
        }
      } else renderState(body, { kind: comparison.running ? "loading" : "empty", title: comparison.running ? "Waiting for this model…" : "No comparison result", description: comparison.running ? "Models run one at a time; complete text appears when ready." : "Run a comparison with both registered models. A missing result is never replaced with a sample." });
    }
  }
  update(); const stopResults = session.subscribe(update);
  return () => { stopEditor(); stopResults(); };
}
return { createComparison, mountComparison };
})();

// Navigation
const navigation = (() => {
if (typeof document !== "undefined" && document.querySelector("#generate-page")) {
const { mountInsights } = insights_page;
const { renderState } = components;
const { createSession } = session_state;
const { mountGeneratePage } = generate_page;
const { createGenerationClient } = generation_client;
const pages = new Map([
  ["generate", "Generate"],
  ["insights", "Model insights"], ["architecture", "Architecture"],
]);
const main = document.querySelector("main");
const generationModel = "transformer-story-continuation-v2";
const session = createSession({ fixedModel: generationModel });
const client = createGenerationClient(session);
let unmount = () => {};

function navigate(moveFocus = false) {
  if (location.hash === "#main-content" && main.childElementCount) { main.focus(); return; }
  unmount();
  unmount = () => {};
  const route = location.hash === "#main-content" ? "generate" : location.hash.replace(/^#\/?/, "") || "generate";
  if (route === "compare") { location.replace("#/generate"); return; }
  document.querySelectorAll("[data-route]").forEach(link => {
    if (link.dataset.route === route) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  if (!pages.has(route)) {
    document.title = "Page not found · Transformer";
    renderState(main, { kind: "error", title: "This page is not part of the workspace.",
      description: "Return to Generate to continue writing.",
      action: { label: "Go to Generate", onClick: () => { location.hash = "/generate"; } },
    });
  } else {
    document.title = `${pages.get(route)} · Transformer`;
    main.replaceChildren(document.querySelector(`#${route}-page`).content.cloneNode(true));
    if (route === "insights" || route === "architecture") unmount = mountInsights(main, { architectureOnly: route === "architecture", architectureModelId: generationModel });
    if (route === "generate") unmount = mountGeneratePage(main, session, { onRetry: client.refresh, onGenerate: client.submit });
  }
  if (moveFocus) { main.focus({ preventScroll: true }); window.scrollTo(0, 0); }
}

window.addEventListener("hashchange", () => navigate(true));
navigate();
client.refresh();

}
return {  };
})();

export { analytics_charts, story_charts_data, story_charts, insights_page, session_history, generation_metadata, session_state, generation_client, model_picker, components, story_output, generate_page, comparison_page, navigation };

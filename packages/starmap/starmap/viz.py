"""Self-contained sci-fi HTML renderer for the constellation view.

The renderer writes a single HTML file with no external dependencies: no CDN,
no web fonts, no network requests. Everything (the starfield, the force-directed
graph, the density heat-map, the forecast fan chart) is drawn with hand-written
Canvas 2D code so the file opens instantly from disk and works offline.

All numbers rendered come from the payload that the caller assembled from real
measurements. The renderer never invents data; where a section has no data it
says so explicitly rather than drawing an empty decorative panel.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
PALETTE = {
    "bg": "#04070f",
    "bg_soft": "#080d1a",
    "panel": "rgba(12,20,38,0.72)",
    "panel_line": "rgba(90,214,255,0.22)",
    "text": "#dbe9ff",
    "text_dim": "#7f92b8",
    "text_faint": "#4d5c78",
    "accent": "#5ad7ff",
    "accent2": "#9d7bff",
    "hot": "#ff6bd6",
    "warm": "#ffd166",
    "alert": "#ff8a5c",
    "good": "#5bffa0",
}

CSS = f"""
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{background:{PALETTE['bg']};color:{PALETTE['text']};
 font-family:"SF Mono",ui-monospace,"PingFang SC","Microsoft YaHei",Menlo,monospace;
 -webkit-font-smoothing:antialiased}}
.wrap{{max-width:1680px;margin:0 auto;padding:20px 22px 60px}}
a{{color:{PALETTE['accent']}}}
header{{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-end;
 justify-content:space-between;padding-bottom:14px;margin-bottom:16px;
 border-bottom:1px solid rgba(90,214,255,0.18)}}
h1{{font-size:19px;font-weight:500;letter-spacing:.06em}}
.sub{{font-size:12px;color:{PALETTE['text_dim']};margin-top:6px;line-height:1.7}}
.badges{{display:flex;flex-wrap:wrap;gap:7px;margin-top:10px}}
.badge{{font-size:11px;padding:3px 9px;border-radius:999px;
 border:1px solid rgba(90,214,255,0.28);color:{PALETTE['text_dim']};
 background:rgba(90,214,255,0.06);white-space:nowrap}}
.badge.on{{border-color:rgba(91,255,160,.45);color:{PALETTE['good']};
 background:rgba(91,255,160,.08)}}
.badge.off{{border-color:rgba(255,107,214,.35);color:{PALETTE['hot']};
 background:rgba(255,107,214,.07)}}
.grid{{display:grid;grid-template-columns:minmax(0,1.62fr) minmax(0,1fr);
 gap:16px;align-items:start}}
@media(max-width:1100px){{.grid{{grid-template-columns:minmax(0,1fr)}}}}
.card{{background:{PALETTE['panel']};border:1px solid {PALETTE['panel_line']};
 border-radius:14px;padding:14px 15px 15px;backdrop-filter:blur(3px)}}
.card h2{{font-size:13px;font-weight:500;letter-spacing:.09em;
 color:{PALETTE['accent']};margin-bottom:3px}}
.card .cap{{font-size:11px;color:{PALETTE['text_faint']};margin-bottom:11px;
 line-height:1.65}}
canvas{{display:block;width:100%;border-radius:9px;background:
 radial-gradient(circle at 50% 42%,#0a1428 0%,#050912 68%,#03060d 100%)}}
#starfield{{position:fixed;inset:0;z-index:0;pointer-events:none}}
.stage{{position:relative;z-index:1}}
.legend{{display:flex;flex-wrap:wrap;gap:12px;margin-top:11px;font-size:11px;
 color:{PALETTE['text_dim']}}}
.legend i{{display:inline-block;width:9px;height:9px;border-radius:50%;
 margin-right:5px;vertical-align:middle}}
.row{{display:flex;gap:16px;flex-wrap:wrap}}
.row>*{{flex:1 1 220px;min-width:0}}
.kv{{display:flex;justify-content:space-between;gap:10px;font-size:12px;
 padding:6px 0;border-bottom:1px solid rgba(255,255,255,.05)}}
.kv:last-child{{border-bottom:none}}
.kv span:first-child{{color:{PALETTE['text_dim']}}}
.kv span:last-child{{color:{PALETTE['text']};text-align:right}}
.ctrl{{display:flex;flex-wrap:wrap;gap:9px;align-items:center;margin-bottom:12px}}
select,input[type=range]{{font-family:inherit;font-size:12px;
 background:rgba(90,214,255,.07);color:{PALETTE['text']};
 border:1px solid rgba(90,214,255,.25);border-radius:7px;padding:5px 9px}}
input[type=range]{{padding:0;height:22px;accent-color:{PALETTE['accent']}}}
label.sw{{display:flex;align-items:center;gap:6px;font-size:11px;
 color:{PALETTE['text_dim']};cursor:pointer;user-select:none}}
label.sw input{{accent-color:{PALETTE['accent']}}}
table{{width:100%;border-collapse:collapse;font-size:12px;table-layout:fixed}}
th,td{{padding:6px 7px;text-align:right;border-bottom:1px solid rgba(255,255,255,.05)}}
th{{color:{PALETTE['text_faint']};font-weight:400;font-size:11px;
 text-align:right}}
th:first-child,td:first-child{{text-align:left}}
.pos{{color:{PALETTE['hot']}}}.neg{{color:{PALETTE['good']}}}
.note{{font-size:11px;color:{PALETTE['text_faint']};line-height:1.75;
 margin-top:10px}}
.hint{{font-size:11px;color:{PALETTE['text_faint']};margin-top:8px;
 display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}}
.tag{{font-size:10px;padding:2px 7px;border-radius:5px;
 background:rgba(157,123,255,.14);color:{PALETTE['accent2']};
 border:1px solid rgba(157,123,255,.3)}}
"""


def _json(data: Any) -> str:
    """Serialise payload safely inside a <script> block.

    ``json.dumps`` happily emits a literal ``</script>`` when a value contains
    one. Inside an HTML ``<script>`` element the parser ends the block at the
    first ``</script>``, so such a value would truncate the payload and let the
    remainder of the string be parsed as markup. Escaping ``<``, ``>`` and
    ``&`` as ``\\uXXXX`` escapes keeps the JSON valid, keeps the decoded value
    byte-identical, and makes the breakout impossible.
    """
    raw = json.dumps(data, ensure_ascii=False, allow_nan=False, default=_safe_default)
    return (
        raw.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _safe_default(obj: Any) -> Any:
    try:
        import math
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
    except Exception:  # noqa: BLE001
        pass
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:  # noqa: BLE001
            pass
    return str(obj)


# --------------------------------------------------------------------------- #
# The renderer
# --------------------------------------------------------------------------- #
def render_html(payload: dict[str, Any], *, title: str = "美林时钟星图") -> str:
    """Return the full standalone HTML document for a constellation payload."""
    meta = payload.get("meta") or {}
    headline = meta.get("title") or title
    generated = meta.get("generated_at", "")

    graph = payload.get("graph") or {"nodes": [], "links": []}
    clouds = payload.get("clouds") or {}
    forecasts = payload.get("forecasts") or {}
    legend = payload.get("legend") or []

    legend_html = "".join(
        f'<span><i style="background:{item.get("color", "#888")}"></i>'
        f'{_esc(item.get("label", ""))}</span>'
        for item in legend
    )

    badges = "".join(
        f'<span class="badge {cls}">{_esc(text)}</span>'
        for text, cls in (payload.get("badges") or [])
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(headline)}</title>
<style>{CSS}</style>
</head>
<body>
<canvas id="starfield"></canvas>
<div class="wrap stage">
<header>
  <div>
    <h1>{_esc(headline)}</h1>
    <div class="sub">{_esc(meta.get("subtitle", ""))}</div>
    <div class="badges">{badges}</div>
  </div>
  <div class="sub" style="text-align:right">
    <div>生成时间 {_esc(generated)}</div>
    <div id="graphStats"></div>
  </div>
</header>

<div class="grid">
  <div class="card">
    <h2>相位星图 · constellation graph</h2>
    <div class="cap">节点为真实测量值驱动：相位概率、资产权重、因子 IC、标的快照、数据来源。连线表示因果、配置或血缘关系。可拖拽节点，点击查看明细。</div>
    <div class="ctrl">
      <label class="sw"><input type="checkbox" class="ly" value="macro_phase" checked>相位</label>
      <label class="sw"><input type="checkbox" class="ly" value="asset" checked>资产</label>
      <label class="sw"><input type="checkbox" class="ly" value="factor" checked>因子</label>
      <label class="sw"><input type="checkbox" class="ly" value="instrument" checked>标的</label>
      <label class="sw"><input type="checkbox" class="ly" value="source">数据源</label>
      <label class="sw"><input type="checkbox" class="ly" value="horizon" checked>预测</label>
      <span class="hint" style="margin:0">
        <label class="sw">连线
          <select id="linkMode">
            <option value="all">全部</option>
            <option value="phase_alloc">配置</option>
            <option value="factor_load">因子暴露</option>
            <option value="data_lineage">血缘</option>
            <option value="phase_transition">相位转移</option>
          </select>
        </label>
      </span>
    </div>
    <canvas id="graph" height="620"></canvas>
    <div class="legend">{legend_html}</div>
    <div class="hint"><span id="tip"></span><span>拖拽节点 · 滚轮缩放 · 点击查看明细</span></div>
  </div>

  <div>
    <div class="card" style="margin-bottom:16px">
      <h2>相位概率与资产权重</h2>
      <div class="cap" id="phaseCap">美林时钟四相位概率与对应资产配置</div>
      <div id="phaseBox"></div>
    </div>
    <div class="card" style="margin-bottom:16px">
      <h2>标的快照</h2>
      <div class="cap" id="instCap">当前研究标的的实测指标</div>
      <div id="instBox"></div>
    </div>
    <div class="card">
      <h2>因子有效性</h2>
      <div class="cap">横截面检验实测 IC / ICIR / 胜率</div>
      <div id="factorBox"></div>
    </div>
  </div>
</div>

<div class="row" style="margin-top:16px">
  <div class="card">
    <h2>相位投影 · phase path</h2>
    <div class="cap" id="pathCap">基于历史相位转移频率的概率外推</div>
    <canvas id="phasepath" height="250"></canvas>
  </div>
  <div class="card">
    <h2>路径扇形图 · bootstrap fan</h2>
    <div class="cap" id="fanCap">历史月度收益自助抽样的分位带</div>
    <canvas id="fan" height="250"></canvas>
  </div>
  <div class="card">
    <h2>密度云图 · density cloud</h2>
    <div class="cap" id="cloudCap">因子空间的核平滑密度</div>
    <canvas id="cloud" height="250"></canvas>
  </div>
</div>

<div class="card" style="margin-top:16px">
  <h2>数据源健康度 · source health</h2>
  <div class="cap">后台数据中枢当前可用的上游；不可用者标注原因而非静默隐藏</div>
  <div id="sourceBox"></div>
</div>

<div class="card" style="margin-top:16px">
  <h2>口径与限制</h2>
  <div id="caveatBox" class="note"></div>
</div>
</div>

<script id="payload" type="application/json">{_json(payload)}</script>
<script>
{_JS}
</script>
</body>
</html>
"""


def _esc(value: Any) -> str:
    text = str(value if value is not None else "")
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))


# --------------------------------------------------------------------------- #
# Client-side JavaScript (no external dependencies)
# --------------------------------------------------------------------------- #
_JS = r"""
const P = JSON.parse(document.getElementById('payload').textContent);
const C = {bg:'#04070f',accent:'#5ad7ff',accent2:'#9d7bff',hot:'#ff6bd6',
  warm:'#ffd166',alert:'#ff8a5c',good:'#5bffa0',text:'#dbe9ff',
  dim:'#7f92b8',faint:'#4d5c78'};

const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const fmt=(v,d)=>(v===null||v===undefined||Number.isNaN(v))?'--':
  Number(v).toFixed(d===undefined?3:d);
const pct=(v,d)=>(v===null||v===undefined||Number.isNaN(v))?'--':
  (Number(v)*100).toFixed(d===undefined?1:d)+'%';

/* ---------- background starfield ---------- */
(function(){
  const cv=document.getElementById('starfield'),ctx=cv.getContext('2d');
  let stars=[],w=0,h=0;
  function resize(){
    w=cv.width=window.innerWidth;h=cv.height=window.innerHeight;
    const n=Math.min(340,Math.floor(w*h/9000));
    stars=Array.from({length:n},()=>({
      x:Math.random()*w,y:Math.random()*h,z:Math.random()*0.85+0.15,
      r:Math.random()*1.25+0.25,tw:Math.random()*0.02+0.004,
      ph:Math.random()*6.28}));
  }
  function tick(){
    ctx.clearRect(0,0,w,h);
    for(const s of stars){
      s.ph+=s.tw;
      const a=(0.16+0.5*s.z)*Math.abs(Math.sin(s.ph));
      ctx.beginPath();
      ctx.arc(s.x,s.y,s.r*s.z,0,6.283);
      ctx.fillStyle='rgba(150,200,255,'+a.toFixed(3)+')';
      ctx.fill();
    }
    requestAnimationFrame(tick);
  }
  resize();tick();window.addEventListener('resize',resize);
})();

/* ---------- force-directed constellation ---------- */
const G=(P.graph||{nodes:[],links:[]});
const gc=document.getElementById('graph'),gx=gc.getContext('2d');
let GW=900,GH=620,DPR=Math.min(2,window.devicePixelRatio||1);
let nodes=[],links=[],hover=null,drag=null,scale=1,offX=0,offY=0;
const LAYER_LABEL={macro_phase:'相位',asset:'资产',factor:'因子',
  instrument:'标的',source:'数据源',horizon:'预测'};

function initGraph(){
  const rect=gc.getBoundingClientRect();
  GW=Math.max(320,rect.width);GH=620;
  gc.width=GW*DPR;gc.height=GH*DPR;gc.style.height=GH+'px';
  gx.setTransform(DPR,0,0,DPR,0,0);
  const cx=GW/2,cy=GH/2;

  // Lay the semantic layers out on concentric rings: the instrument sits in
  // the centre, factors orbit it, assets and phases sit outside, data sources
  // form the outer halo. Seeding positions on rings (rather than a single
  // random cloud) is what keeps the graph legible.
  const RING={instrument:0.0,factor:0.30,asset:0.50,macro_phase:0.66,
    horizon:0.82,source:1.0};
  const byKind={};
  nodes=(G.nodes||[]).map((n)=>{
    const k=n.kind;byKind[k]=(byKind[k]||0)+1;
    const total=byKind[k];
    const ring=RING[k]!==undefined?RING[k]:0.7;
    const rad=Math.min(GW,GH)*0.44*ring;
    const ang=(byKind[k]/Math.max(1,total))*6.283 + (k==='source'?0.4:0);
    return Object.assign({},n,{x:cx+Math.cos(ang)*rad,y:cy+Math.sin(ang)*rad,
      vx:0,vy:0,pinned:false});
  });
  links=(G.links||[]).map(l=>({source:l.source,target:l.target,
    kind:l.kind,weight:l.weight,detail:l.detail}));
  for(let i=0;i<420;i++)tickPhysics(0.016,false);
}
function activeLayers(){
  return Array.from(document.querySelectorAll('.ly')).filter(c=>c.checked)
    .map(c=>c.value);
}
function visible(n){
  return activeLayers().indexOf(n.kind)>=0;
}
function tickPhysics(step,render){
  const idx={};nodes.forEach((n,i)=>idx[n.id]=i);
  const shown=nodes.filter(visible);
  const cx=GW/2,cy=GH/2;
  const RING={instrument:0.0,factor:0.30,asset:0.50,macro_phase:0.66,
    horizon:0.82,source:1.0};

  // 1) Repulsion between every visible pair.
  for(let i=0;i<shown.length;i++){
    const a=shown[i];
    for(let j=i+1;j<shown.length;j++){
      const b=shown[j];
      let dx=b.x-a.x,dy=b.y-a.y;
      let d2=dx*dx+dy*dy;
      if(d2<0.01){dx=(Math.random()-.5)*2;dy=(Math.random()-.5)*2;d2=1;}
      const d=Math.sqrt(d2);
      const minD=(a.size||6)+(b.size||6)+26;
      let f=0;
      if(d<minD) f=0.9*(minD-d)/minD;          // hard separation
      else f=0.10;                              // gentle global spread
      const fx=dx/d*f,fy=dy/d*f;
      if(!a.pinned){a.vx-=fx*step*60;a.vy-=fy*step*60;}
      if(!b.pinned){b.vx+=fx*step*60;b.vy+=fy*step*60;}
    }
  }
  // 2) Link springs (weak, so they guide rather than collapse the layout).
  const lm=document.getElementById('linkMode').value;
  for(const l of links){
    if(lm!=='all'&&l.kind!==lm)continue;
    const a=nodes[idx[l.source]],b=nodes[idx[l.target]];
    if(!a||!b)continue;
    if(!visible(a)&&!visible(b))continue;
    const dx=b.x-a.x,dy=b.y-a.y;
    const d=Math.max(1,Math.hypot(dx,dy));
    const rest=l.kind==='data_lineage'?260:190;
    const k=0.006;
    const f=(d-rest)*k;
    const fx=dx/d*f,fy=dy/d*f;
    if(!a.pinned){a.vx+=fx*step*60;a.vy+=fy*step*60;}
    if(!b.pinned){b.vx-=fx*step*60;b.vy-=fy*step*60;}
  }
  // 3) Ring gravity keeps the semantic layering visible.
  for(const n of nodes){
    if(n.pinned)continue;
    const ring=RING[n.kind]!==undefined?RING[n.kind]:0.6;
    const target=Math.min(GW,GH)*0.44*ring;
    const dx=n.x-cx,dy=n.y-cy;
    const d=Math.max(1,Math.hypot(dx,dy));
    const f=(d-target)*0.010;
    n.vx-=dx/d*f*step*60;n.vy-=dy/d*f*step*60;
    n.vx*=0.90;n.vy*=0.90;
    n.x+=n.vx*step*60;n.y+=n.vy*step*60;
    const pad=(n.size||6)+14;
    n.x=clamp(n.x,pad,GW-pad);n.y=clamp(n.y,pad,GH-pad);
  }
  if(render)drawGraph();
}
function linkColor(kind){
  return {phase_alloc:'rgba(90,214,255,',factor_load:'rgba(255,107,214,',
    data_lineage:'rgba(120,150,190,',phase_transition:'rgba(157,123,255,'}[kind]
    ||'rgba(120,150,190,';
}
function drawGraph(){
  gx.clearRect(0,0,GW,GH);
  gx.save();
  gx.translate(offX,offY);gx.scale(scale,scale);
  const lm=document.getElementById('linkMode').value;
  const idx={};nodes.forEach((n,i)=>idx[n.id]=i);
  // Lineage links are drawn first (faintest, lowest) so the semantic links
  // stay legible on top of them.
  const order=['data_lineage','phase_transition','phase_alloc','factor_load'];
  for(const kind of order){
    if(lm!=='all'&&lm!==kind)continue;
    for(const l of links){
      if(l.kind!==kind)continue;
      const a=nodes[idx[l.source]],b=nodes[idx[l.target]];
      if(!a||!b)continue;
      if(!visible(a)&&!visible(b))continue;
      const alpha=kind==='data_lineage'?0.10:
        clamp(Math.abs(l.weight)*0.7+0.16,0.14,0.62);
      gx.strokeStyle=linkColor(kind)+alpha+')';
      gx.lineWidth=kind==='data_lineage'?0.6:
        clamp(Math.abs(l.weight)*1.4,0.7,3.0);
      gx.beginPath();
      if(kind==='data_lineage'){
        const mx=(a.x+b.x)/2,my=(a.y+b.y)/2;
        const nx=-(b.y-a.y)*0.18,ny=(b.x-a.x)*0.18;
        gx.moveTo(a.x,a.y);gx.quadraticCurveTo(mx+nx,my+ny,b.x,b.y);
      }else{
        gx.moveTo(a.x,a.y);gx.lineTo(b.x,b.y);
      }
      gx.stroke();
    }
  }
  for(const n of nodes){
    if(!visible(n))continue;
    const r=n.size||6;
    const isHover=hover&&hover.id===n.id;
    gx.beginPath();gx.arc(n.x,n.y,r+5,0,6.283);
    gx.fillStyle=n.color+'1f';gx.fill();
    gx.beginPath();gx.arc(n.x,n.y,r,0,6.283);
    gx.fillStyle=n.color+'33';gx.fill();
    gx.beginPath();gx.arc(n.x,n.y,r*0.55,0,6.283);
    gx.fillStyle=n.color;gx.fill();
    if(isHover){
      gx.strokeStyle=C.text;gx.lineWidth=1.2;
      gx.beginPath();gx.arc(n.x,n.y,r+7,0,6.283);gx.stroke();
    }
    if(scale>0.62){
      gx.font='11px ui-monospace,monospace';
      gx.textAlign='center';
      const label=n.label;
      const w=gx.measureText(label).width;
      gx.fillStyle='rgba(4,7,15,0.72)';
      gx.fillRect(n.x-w/2-3,n.y+r+3,w+6,13);
      gx.fillStyle=isHover?C.text:C.dim;
      gx.fillText(label,n.x,n.y+r+13);
    }
  }
  gx.restore();
  const st=document.getElementById('graphStats');
  const vis=nodes.filter(visible).length;
  if(st)st.textContent='节点 '+vis+'/'+nodes.length+' · 连线 '+links.length;
}
gc.addEventListener('mousemove',e=>{
  const rect=gc.getBoundingClientRect();
  const mx=(e.clientX-rect.left-offX)/scale,my=(e.clientY-rect.top-offY)/scale;
  if(drag){
    const n=drag;n.pinned=true;n.x=mx;n.y=my;n.vx=0;n.vy=0;return;
  }
  let found=null;
  for(const n of nodes){
    if(!visible(n))continue;
    if(Math.hypot(n.x-mx,n.y-my)<(n.size||6)+7){found=n;break;}
  }
  hover=found;
  const tip=document.getElementById('tip');
  if(tip){
    if(found){
      const m=found.metrics||{};
      const ms=Object.keys(m).slice(0,6).map(k=>k+'='+fmt(m[k],3)).join(' · ');
      tip.textContent=found.label+' — '+(found.detail||'')+(ms?' · '+ms:'');
    }else tip.textContent='';
  }
  gc.style.cursor=found?'pointer':'default';
  drawGraph();
});
gc.addEventListener('mousedown',e=>{
  const rect=gc.getBoundingClientRect();
  const mx=(e.clientX-rect.left-offX)/scale,my=(e.clientY-rect.top-offY)/scale;
  for(const n of nodes){
    if(!visible(n))continue;
    if(Math.hypot(n.x-mx,n.y-my)<(n.size||6)+7){drag=n;n.pinned=true;break;}
  }
});
window.addEventListener('mouseup',()=>{
  if(drag){drag.pinned=false;drag=null;}
});
gc.addEventListener('wheel',e=>{
  e.preventDefault();
  const rect=gc.getBoundingClientRect();
  const mx=e.clientX-rect.left,my=e.clientY-rect.top;
  const f=e.deltaY<0?1.12:1/1.12;
  const ns=clamp(scale*f,0.45,3.2);
  offX=mx-(mx-offX)*(ns/scale);offY=my-(my-offY)*(ns/scale);
  scale=ns;drawGraph();
},{passive:false});

/* ---------- phase path chart ---------- */
function drawPhasePath(){
  const cv=document.getElementById('phasepath');if(!cv)return;
  const rect=cv.getBoundingClientRect(),W=Math.max(260,rect.width),H=250;
  cv.width=W*DPR;cv.height=H*DPR;cv.style.height=H+'px';
  const ctx=cv.getContext('2d');ctx.setTransform(DPR,0,0,DPR,0,0);
  ctx.clearRect(0,0,W,H);
  const fc=(P.forecasts||{}).phase_path||{};
  if(!fc.available){
    ctx.fillStyle=C.faint;ctx.font='12px ui-monospace,monospace';
    ctx.textAlign='center';
    ctx.fillText(fc.note||'没有可用的相位投影',W/2,H/2);return;
  }
  const steps=fc.steps||[];
  const phases=[...new Set(steps.flatMap(s=>Object.keys(s.distribution||{})))];
  if(!phases.length)return;
  const cols={reflation:C.hot,recovery:C.good,overheat:C.alert,contraction:C.accent2,
    stagflation:C.warm};
  const PL=34,PR=10,PT=14,PB=26;
  const cw=(W-PL-PR)/Math.max(1,steps.length);
  for(let i=0;i<steps.length;i++){
    const s=steps[i];const x=PL+i*cw;
    let acc=0;
    for(const p of phases){
      const v=(s.distribution||{})[p]||0;
      const h=v*(H-PT-PB);
      ctx.fillStyle=(cols[p]||C.accent)+'44';
      ctx.fillRect(x+3,PT+acc,Math.max(3,cw-6),h);
      acc+=h;
    }
    ctx.fillStyle=C.faint;ctx.font='10px ui-monospace,monospace';
    ctx.textAlign='center';
    ctx.fillText('T+'+s.step,x+cw/2,H-9);
  }
  ctx.fillStyle=C.dim;ctx.font='10px ui-monospace,monospace';ctx.textAlign='left';
  ctx.fillText(fc.current_phase||'',PL,PT-3);
  const lw=18;
  let lx=PL;
  for(const p of phases){
    ctx.fillStyle=cols[p]||C.accent;
    ctx.fillRect(lx,H-2,10,2);
    ctx.fillStyle=C.faint;ctx.textAlign='left';
    ctx.fillText(p,lx+13,H-1);
    lx+=ctx.measureText(p).width+34;
  }
  const cp=document.getElementById('pathCap');
  if(cp&&fc.basis)cp.textContent=fc.basis;
}

/* ---------- bootstrap fan ---------- */
function drawFan(){
  const cv=document.getElementById('fan');if(!cv)return;
  const rect=cv.getBoundingClientRect(),W=Math.max(260,rect.width),H=250;
  cv.width=W*DPR;cv.height=H*DPR;cv.style.height=H+'px';
  const ctx=cv.getContext('2d');ctx.setTransform(DPR,0,0,DPR,0,0);
  ctx.clearRect(0,0,W,H);
  const fc=(P.forecasts||{}).price_path||{};
  if(!fc.available){
    ctx.fillStyle=C.faint;ctx.font='12px ui-monospace,monospace';
    ctx.textAlign='center';
    ctx.fillText(fc.note||'没有可用的路径投影',W/2,H/2);return;
  }
  const st=fc.steps||[];if(!st.length)return;
  const PL=8,PR=8,PT=16,PB=22;
  const vals=st.flatMap(s=>[s.lower,s.upper]);
  const lo=Math.min.apply(null,vals),hi=Math.max.apply(null,vals);
  const pad=(hi-lo)*0.08||1;
  const ymin=lo-pad,ymax=hi+pad;
  const X=i=>PL+i*(W-PL-PR)/Math.max(1,st.length-1);
  const Y=v=>PT+(1-(v-ymin)/(ymax-ymin))*(H-PT-PB);
  ctx.beginPath();
  st.forEach((s,i)=>i?ctx.lineTo(X(i),Y(s.upper)):ctx.moveTo(X(i),Y(s.upper)));
  for(let i=st.length-1;i>=0;i--)ctx.lineTo(X(i),Y(st[i].lower));
  ctx.closePath();
  const grd=ctx.createLinearGradient(0,PT,0,H-PB);
  grd.addColorStop(0,'rgba(90,214,255,0.34)');
  grd.addColorStop(1,'rgba(90,214,255,0.03)');
  ctx.fillStyle=grd;ctx.fill();
  ctx.beginPath();
  st.forEach((s,i)=>i?ctx.lineTo(X(i),Y(s.median)):ctx.moveTo(X(i),Y(s.median)));
  ctx.strokeStyle=C.accent;ctx.lineWidth=1.6;ctx.stroke();
  ctx.beginPath();
  st.forEach((s,i)=>i?ctx.lineTo(X(i),Y(s.upper)):ctx.moveTo(X(i),Y(s.upper)));
  ctx.strokeStyle='rgba(90,214,255,0.5)';ctx.lineWidth=0.8;ctx.stroke();
  ctx.beginPath();
  st.forEach((s,i)=>i?ctx.lineTo(X(i),Y(s.lower)):ctx.moveTo(X(i),Y(s.lower)));
  ctx.strokeStyle='rgba(90,214,255,0.5)';ctx.lineWidth=0.8;ctx.stroke();
  const last=st[st.length-1];
  ctx.fillStyle=C.warm;ctx.font='10px ui-monospace,monospace';ctx.textAlign='right';
  ctx.fillText('中位 '+fmt(last.median,2)+' · 上行概率 '+pct(last.prob_up,0),
    W-PR,PT-4);
  const cp=document.getElementById('fanCap');
  if(cp&&fc.basis)cp.textContent=fc.basis;
}

/* ---------- density cloud ---------- */
function drawCloud(){
  const cv=document.getElementById('cloud');if(!cv)return;
  const rect=cv.getBoundingClientRect(),W=Math.max(260,rect.width),H=250;
  cv.width=W*DPR;cv.height=H*DPR;cv.style.height=H+'px';
  const ctx=cv.getContext('2d');ctx.setTransform(DPR,0,0,DPR,0,0);
  ctx.clearRect(0,0,W,H);
  const cl=(P.clouds||{}).correlation||{};
  if(!cl.available){
    ctx.fillStyle=C.faint;ctx.font='12px ui-monospace,monospace';
    ctx.textAlign='center';
    ctx.fillText(cl.note||'没有可用的密度云图',W/2,H/2);return;
  }
  const dens=cl.density||[];
  if(!dens.length)return;
  const dh=dens.length,dw=dens[0].length;
  const cw=W/dw,ch=H/dh;
  for(let r=0;r<dh;r++){
    for(let c=0;c<dw;c++){
      const v=dens[r][c];
      if(v<=0.002)continue;
      const t=Math.pow(v,0.55);
      const rr=Math.round(90+t*165),gg=Math.round(215-t*140),bb=Math.round(255-t*40);
      // Alpha falls off with density so the low-density halo fades out
      // instead of ending in a hard rectangular edge.
      const a=(0.04+t*0.62)*Math.min(1,t*2.2);
      ctx.fillStyle='rgba('+rr+','+gg+','+bb+','+a.toFixed(3)+')';
      ctx.fillRect(c*cw-0.5,H-(r+1)*ch-0.5,cw+1,ch+1);
    }
  }
  // Soft radial vignette to blend the grid boundary into the background.
  const vg=ctx.createRadialGradient(W/2,H/2,Math.min(W,H)*0.18,W/2,H/2,Math.max(W,H)*0.62);
  vg.addColorStop(0,'rgba(4,7,15,0)');
  vg.addColorStop(1,'rgba(4,7,15,0.92)');
  ctx.fillStyle=vg;ctx.fillRect(0,0,W,H);
  for(const pt of (cl.points||[]).slice(0,320)){
    const x=cl.x_range?((pt[0]-cl.x_range[0])/(cl.x_range[1]-cl.x_range[0]))*W:W/2;
    const y=cl.y_range?(H-((pt[1]-cl.y_range[0])/(cl.y_range[1]-cl.y_range[0]))*H):H/2;
    ctx.fillStyle='rgba(255,255,255,0.42)';
    ctx.fillRect(x-0.9,y-0.9,1.8,1.8);
  }
  ctx.fillStyle=C.faint;ctx.font='10px ui-monospace,monospace';ctx.textAlign='left';
  ctx.fillText(cl.x_label||'',6,H-6);
  ctx.save();ctx.translate(11,6);ctx.rotate(-Math.PI/2);
  ctx.textAlign='right';ctx.fillText(cl.y_label||'',0,0);ctx.restore();
  const cp=document.getElementById('cloudCap');
  if(cp)cp.textContent=(cl.x_label||'')+' × '+(cl.y_label||'')+' · 样本 '+(cl.n_points||0);
}

/* ---------- side panels ---------- */
function esc(s){return String(s==null?'':s).replace(/&/g,'&amp;')
  .replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function renderPhaseBox(){
  const box=document.getElementById('phaseBox');if(!box)return;
  const phases=(P.phases||[]);
  const alloc=(P.allocations||[]);
  if(!phases.length&&!alloc.length){
    box.innerHTML='<div class="note">没有相位数据</div>';return;
  }
  let html='';
  if(phases.length){
    html+='<div style="font-size:11px;color:'+C.faint+';margin-bottom:6px">'+
      '相位概率（软分类，非标定预测概率）</div>';
    html+=phases.map(p=>'<div class="kv"><span>'+esc(p.label)+
      '</span><span>'+pct(p.probability,1)+'</span></div>').join('');
  }
  if(alloc.length){
    html+='<div style="font-size:11px;color:'+C.faint+';margin:12px 0 6px">'+
      '资产目标权重</div>';
    html+=alloc.map(a=>'<div class="kv"><span>'+esc(a.label)+
      '</span><span>'+pct(a.weight,1)+'</span></div>').join('');
  }
  box.innerHTML=html;
}
function renderInstBox(){
  const box=document.getElementById('instBox');if(!box)return;
  const ins=P.instrument||{};
  const rows=(ins.metrics||[]);
  if(!rows.length){
    box.innerHTML='<div class="note">'+esc(ins.note||'没有标的可用快照')+'</div>';
    return;
  }
  box.innerHTML='<div style="font-size:14px;color:'+C.text+';margin-bottom:8px">'+
    esc(ins.symbol||'')+' '+(ins.label?'<span style="color:'+C.dim+
    ';font-size:12px">'+esc(ins.label)+'</span>':'')+'</div>'+
    rows.map(r=>'<div class="kv"><span>'+esc(r[0])+'</span><span>'+
      esc(r[1])+'</span></div>').join('');
}
function renderFactorBox(){
  const box=document.getElementById('factorBox');if(!box)return;
  const fs=(P.factors||[]);
  if(!fs.length){box.innerHTML='<div class="note">没有因子检验结果</div>';return;}
  box.innerHTML='<table><thead><tr><th>因子</th><th>RankIC</th><th>ICIR</th>'+
    '<th>胜率</th><th>N</th></tr></thead><tbody>'+
    fs.map(f=>'<tr><td>'+esc(f.name)+'</td><td class="'+
      (f.ic_mean>=0?'pos':'neg')+'">'+fmt(f.ic_mean,4)+'</td><td>'+
      fmt(f.icir,3)+'</td><td>'+pct(f.positive_ratio,1)+'</td><td>'+
      (f.n_periods||0)+'</td></tr>').join('')+'</tbody></table>';
}
function renderSourceBox(){
  const box=document.getElementById('sourceBox');if(!box)return;
  const ss=(P.sources||[]);
  if(!ss.length){box.innerHTML='<div class="note">没有数据源记录</div>';return;}
  box.innerHTML='<table><thead><tr><th>数据源</th><th>能力</th><th>状态</th>'+
    '<th>说明</th></tr></thead><tbody>'+
    ss.map(s=>'<tr><td>'+esc(s.source_id)+'</td><td>'+esc(s.capabilities||'')+
      '</td><td style="text-align:left;color:'+(s.available?C.good:C.hot)+'">'+
      esc(s.status_detail||'')+'</td><td style="text-align:left;color:'+C.faint+
      ';font-size:11px">'+esc((s.note||'').slice(0,90))+'</td></tr>').join('')+
    '</tbody></table>';
}
function renderCaveats(){
  const box=document.getElementById('caveatBox');if(!box)return;
  const cs=(P.caveats||[]);
  box.innerHTML=cs.length?cs.map(c=>'· '+esc(c)).join('<br>'):
    '<span style="color:'+C.faint+'">未提供口径说明</span>';
}

/* ---------- boot ---------- */
initGraph();
drawGraph();
renderPhaseBox();renderInstBox();renderFactorBox();renderSourceBox();
renderCaveats();drawPhasePath();drawFan();drawCloud();
document.querySelectorAll('.ly').forEach(el=>
  el.addEventListener('change',()=>drawGraph()));
document.getElementById('linkMode').addEventListener('change',()=>drawGraph());
let raf;
function loop(){tickPhysics(0.016,true);raf=requestAnimationFrame(loop);}
loop();
window.addEventListener('resize',()=>{initGraph();drawGraph();
  drawPhasePath();drawFan();drawCloud();});
"""


def write_html(payload: dict[str, Any], target: Path) -> Path:
    """Render the payload to a standalone HTML file and return its path."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html(payload), encoding="utf-8")
    return target

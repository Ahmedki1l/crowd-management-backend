"""Dev/ops tools for the single-node ``--api`` deployment: an ROI (occupancy
zone) editor plus a validation-snapshot capture.

These are convenience helpers, not part of the HLD read model. Heavy deps
(``cv2``, the detector) are imported lazily inside the handlers so the core /
test environment stays light. Endpoints that touch cameras require auth; the ROI
page itself is static HTML that calls the authed endpoints with the user's token.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, Response
from sqlalchemy.orm import Session

from app.api.deps import AuthDep, db_session
from app.api.validation_capture import capture_validation_images, fetch_frame
from app.services.camera_service import CameraService

router = APIRouter(prefix="/tools", tags=["tools"])


@router.get("/cameras/{camera_id}/frame", dependencies=[AuthDep])
def get_frame(camera_id: int, session: Session = Depends(db_session)) -> Response:
    """Return a live JPEG frame from the camera (for drawing an ROI on)."""
    import cv2

    _, frame = fetch_frame(CameraService(session), camera_id)
    if frame is None:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, f"could not fetch a frame from camera {camera_id}"
        )
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "jpeg encode failed")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@router.post("/capture", dependencies=[AuthDep])
def capture(session: Session = Depends(db_session)) -> dict:
    """Capture allowlisted occupancy cameras, annotate zone + detections, save to
    the validation folder (wiping old), and return the count rollup.

    green box = counted (inside a zone), grey = detected but outside, orange =
    zone outline.
    """
    return capture_validation_images(session)


@router.get("/roi", response_class=HTMLResponse, include_in_schema=False)
def roi_editor() -> HTMLResponse:
    """Serve the self-contained ROI (occupancy zone) editor page."""
    return HTMLResponse(_ROI_HTML)


_ROI_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>ROI editor</title>
<style>
  body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee}
  header{padding:10px 14px;background:#1b1b1b;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  input,select,button{font:inherit;padding:6px 8px;border-radius:6px;border:1px solid #444;background:#222;color:#eee}
  button{cursor:pointer;background:#2b6}button.secondary{background:#345}
  #wrap{display:flex;gap:12px;padding:12px}
  #canvas{border:1px solid #444;cursor:crosshair;max-width:100%}
  #side{width:280px;display:flex;flex-direction:column;gap:8px}
  .hint{font-size:12px;color:#aaa}
  #log{white-space:pre-wrap;font-family:monospace;font-size:12px;background:#000;padding:8px;border-radius:6px;min-height:60px;max-height:220px;overflow:auto}
  label{font-size:12px;color:#bbb}
</style></head><body>
<header>
  <strong>ROI editor</strong>
  <input id="token" placeholder="paste Bearer token" size="34">
  <select id="camera"></select>
  <button id="load" class="secondary">Load frame</button>
  <span class="hint">click on the image to add polygon points</span>
</header>
<div id="wrap">
  <canvas id="canvas" width="960" height="540"></canvas>
  <div id="side">
    <label>Mode</label>
    <select id="mode"><option value="zone">zone (occupancy)</option>
      <option value="line">line (entry/exit)</option></select>
    <label>Name</label><input id="zname" value="ROI">
    <label>Type (zone only)</label>
    <select id="ztype"><option>occupancy</option></select>
    <label>dt_space_id (zone space) &mdash; <b>REQUIRED</b>: history is keyed by it</label>
    <input id="zspace" required placeholder="e.g. b1-waiting-area / gf-waiting-area">
    <label>area_id (line entry/exit)</label>
    <input id="larea" placeholder="e.g. main-entrance">
    <div style="display:flex;gap:6px">
      <button id="undo" class="secondary">Undo</button>
      <button id="clear" class="secondary">Clear</button>
    </div>
    <button id="save">Save (zone or line)</button>
    <button id="existing" class="secondary">Show existing zones</button>
    <button id="delzones" class="secondary">Delete existing zones (this camera)</button>
    <button id="capture" class="secondary">Capture validation images</button>
    <div id="log">ready.</div>
  </div>
</div>
<script>
const API = location.origin + '/api/v1';
const $ = id => document.getElementById(id);
const cv = $('canvas'), ctx = cv.getContext('2d');
let img = new Image(), scale = 1, pts = [], existing = [], existingLines = [];
$('token').value = localStorage.getItem('roi_token') || '';
function log(m){ $('log').textContent = (typeof m==='string'?m:JSON.stringify(m,null,2)); }
function hdr(){ const t=$('token').value.trim(); localStorage.setItem('roi_token',t);
  return {'Authorization':'Bearer '+t}; }

async function loadCameras(){
  try{
    const r = await fetch(API+'/cameras',{headers:hdr()});
    if(!r.ok){ log('cameras '+r.status+' — check token'); return; }
    const cams = await r.json();
    $('camera').innerHTML = cams.map(c=>`<option value="${c.id}">${c.id} · ${c.name} (${c.ip})</option>`).join('');
  }catch(e){ log(String(e)); }
}
function draw(){
  ctx.clearRect(0,0,cv.width,cv.height);
  if(img.width){ ctx.drawImage(img,0,0,cv.width,cv.height); }
  ctx.strokeStyle='#ffb020'; ctx.lineWidth=2;
  for(const z of existing){ poly(z.polygon,'#ffb02066'); }
  if($('mode').value==='line'){ drawLine(); return; }
  // current polygon (green)
  ctx.strokeStyle='#22dd66'; ctx.fillStyle='#22dd6633';
  if(pts.length){
    ctx.beginPath(); ctx.moveTo(pts[0][0]*scale,pts[0][1]*scale);
    for(const p of pts.slice(1)) ctx.lineTo(p[0]*scale,p[1]*scale);
    if(pts.length>2){ ctx.closePath(); ctx.fill(); } ctx.stroke();
    for(const p of pts){ ctx.fillStyle='#22dd66'; ctx.beginPath(); ctx.arc(p[0]*scale,p[1]*scale,4,0,7); ctx.fill(); }
  }
}
function drawLine(){
  // existing lines (blue) with their IN normal
  ctx.lineWidth=2;
  for(const l of existingLines){
    const a=l.points[0], b=l.points[1]; ctx.strokeStyle='#4aa3ff';
    ctx.beginPath(); ctx.moveTo(a[0]*scale,a[1]*scale); ctx.lineTo(b[0]*scale,b[1]*scale); ctx.stroke();
    const mx=(a[0]+b[0])/2, my=(a[1]+b[1])/2, d=l.in_direction;
    ctx.strokeStyle='#88ccff'; ctx.beginPath();
    ctx.moveTo(mx*scale,my*scale); ctx.lineTo((mx+d[0]*60)*scale,(my+d[1]*60)*scale); ctx.stroke();
  }
  // pts[0],pts[1] = the crossing line (red); pts[2] = a point on the IN side (green arrow)
  if(pts.length>=2){
    ctx.strokeStyle='#ff3b3b'; ctx.lineWidth=4; ctx.beginPath();
    ctx.moveTo(pts[0][0]*scale,pts[0][1]*scale); ctx.lineTo(pts[1][0]*scale,pts[1][1]*scale); ctx.stroke();
    if(pts.length===3){
      const mx=(pts[0][0]+pts[1][0])/2*scale, my=(pts[0][1]+pts[1][1])/2*scale;
      ctx.strokeStyle='#22dd66'; ctx.lineWidth=3;
      ctx.beginPath(); ctx.moveTo(mx,my); ctx.lineTo(pts[2][0]*scale,pts[2][1]*scale); ctx.stroke();
      ctx.fillStyle='#22dd66'; ctx.beginPath(); ctx.arc(pts[2][0]*scale,pts[2][1]*scale,7,0,7); ctx.fill();
    }
  }
  ctx.fillStyle='#ff3b3b';
  for(const p of pts){ ctx.beginPath(); ctx.arc(p[0]*scale,p[1]*scale,5,0,7); ctx.fill(); }
}
function poly(points,fill){ ctx.beginPath(); ctx.moveTo(points[0][0]*scale,points[0][1]*scale);
  for(const p of points.slice(1)) ctx.lineTo(p[0]*scale,p[1]*scale); ctx.closePath();
  if(fill){ctx.fillStyle=fill; ctx.fill();} ctx.stroke(); }

$('load').onclick = async ()=>{
  const id = $('camera').value; existing=[]; pts=[];
  try{
    const r = await fetch(API+'/tools/cameras/'+id+'/frame',{headers:hdr()});
    if(!r.ok){ log('frame '+r.status); return; }
    const blob = await r.blob(); const url = URL.createObjectURL(blob);
    img = new Image();
    img.onload = ()=>{ scale = cv.width/img.width; cv.height = Math.round(img.height*scale);
      draw(); log('frame loaded '+img.width+'x'+img.height+'  scale='+scale.toFixed(3)); };
    img.src = url;
  }catch(e){ log(String(e)); }
};
cv.onmousedown = e=>{
  const rect = cv.getBoundingClientRect();
  const x = (e.clientX-rect.left)*(cv.width/rect.width)/scale;
  const y = (e.clientY-rect.top)*(cv.height/rect.height)/scale;
  if($('mode').value==='line' && pts.length>=3){ log('line ready (Clear to redo)'); return; }
  pts.push([Math.round(x),Math.round(y)]); draw();
  if($('mode').value==='line'){
    log(pts.length<2?'click the other end of the line':
        (pts.length<3?'now click a point on the IN (enter) side':'line + direction set - Save'));
  }
};
$('mode').onchange = ()=>{ pts=[]; draw();
  log($('mode').value==='line'
    ? 'LINE mode: click 2 endpoints across the doorway, then 1 point on the IN side, set area_id, Save'
    : 'ZONE mode: click polygon points, set dt_space_id, Save'); };
$('undo').onclick = ()=>{ pts.pop(); draw(); };
$('clear').onclick = ()=>{ pts=[]; draw(); };
$('existing').onclick = async ()=>{
  const cam = $('camera').value;
  try{
    if($('mode').value==='line'){
      const r = await fetch(API+'/lines',{headers:hdr()}); const ls = await r.json();
      existingLines = ls.filter(l=>String(l.camera_id)===String(cam)); draw();
      log('existing lines: '+(existingLines.map(l=>l.name+' ('+l.area_id+')').join(', ')||'none'));
      return;
    }
    const r = await fetch(API+'/zones',{headers:hdr()}); const zs = await r.json();
    existing = zs.filter(z=>String(z.camera_id)===String(cam)); draw();
    log('existing zones on this camera: '+(existing.map(z=>z.name).join(', ')||'none'));
  }catch(e){ log(String(e)); }
};
$('save').onclick = async ()=>{
  if($('mode').value==='line') return saveLine();
  if(pts.length<3){ log('need at least 3 points'); return; }
  const space = $('zspace').value.trim();
  if(!space){
    log('dt_space_id is required. Occupancy history is stored per space, so a zone\\n'
      + 'without one is counted live and then forgotten - it appears in no history at all.\\n'
      + 'Use an existing space (e.g. b1-waiting-area) so this zone feeds it.');
    return;
  }
  const body = { camera_id: Number($('camera').value), name: $('zname').value,
    type: $('ztype').value, polygon: pts, dt_space_id: space };
  try{
    const r = await fetch(API+'/zones',{method:'POST',
      headers:{...hdr(),'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j = await r.json(); if(!r.ok){ log('save '+r.status+'\\n'+JSON.stringify(j)); return; }
    log('saved zone id '+j.id+' ('+j.name+') with '+j.polygon.length+' points'); $('existing').onclick();
  }catch(e){ log(String(e)); }
};
async function saveLine(){
  if(pts.length<3){ log('line: click 2 endpoints, then 1 point on the IN (enter) side'); return; }
  const area=$('larea').value.trim();
  if(!area){ log('set the line area_id (e.g. main-entrance)'); return; }
  const [p1,p2,p3]=pts;
  let nx=-(p2[1]-p1[1]), ny=(p2[0]-p1[0]);                  // a normal to the line
  const mx=(p1[0]+p2[0])/2, my=(p1[1]+p2[1])/2;
  if((p3[0]-mx)*nx+(p3[1]-my)*ny < 0){ nx=-nx; ny=-ny; }    // orient toward the IN point
  const len=Math.hypot(nx,ny)||1;
  const body={ camera_id:Number($('camera').value), name:$('zname').value, points:[p1,p2],
    in_direction:[Math.round(nx/len*100)/100, Math.round(ny/len*100)/100], area_id:area };
  try{ const r=await fetch(API+'/lines',{method:'POST',
      headers:{...hdr(),'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j=await r.json(); if(!r.ok){ log('save line '+r.status+'\\n'+JSON.stringify(j)); return; }
    log('saved line id '+j.id+' area='+j.area_id+' in_direction=['+j.in_direction+']');
  }catch(e){ log(String(e)); }
}
$('delzones').onclick = async ()=>{
  const cam = $('camera').value;
  try{ const r = await fetch(API+'/zones',{headers:hdr()}); const zs = await r.json();
    const mine = zs.filter(z=>String(z.camera_id)===String(cam));
    if(!mine.length){ log('no existing zones on this camera'); return; }
    if(!confirm('Delete '+mine.length+' existing zone(s) on camera '+cam+'?\\n\\n'
      + 'History is safe: it is keyed by dt_space_id, not by zone id, so redrawing a\\n'
      + 'polygon no longer orphans it. Re-use the SAME dt_space_id to keep the series\\n'
      + 'continuous.')) return;
    for(const z of mine){ await fetch(API+'/zones/'+z.id,{method:'DELETE',headers:hdr()}); }
    existing=[]; draw(); log('deleted '+mine.length+' zone(s). Now draw the new ROI and Save.');
  }catch(e){ log(String(e)); }
};
$('capture').onclick = async ()=>{
  log('capturing...');
  try{ const r = await fetch(API+'/tools/capture',{method:'POST',headers:hdr()}); log(await r.json()); }
  catch(e){ log(String(e)); }
};
$('token').onchange = loadCameras;
loadCameras();
</script></body></html>
"""

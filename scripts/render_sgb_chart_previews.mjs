// Export the default view of the existing public Plotly documents as PNGs.
// Requires Node.js22+ and Chrome; run against a locally served docs directory:
// node scripts/render_sgb_chart_previews.mjs --base-url http://127.0.0.1:8000/docs/images/sgb-concept/ --output-dir docs/images/sgb-concept
import {spawn} from 'node:child_process';
import {createHash} from 'node:crypto';
import {mkdir, mkdtemp, readFile, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join, resolve} from 'node:path';

const args = process.argv.slice(2);
function option(name) {
  const at = args.indexOf(name);
  if (at < 0 || !args[at + 1]) throw new Error(`Missing ${name}`);
  return args[at + 1];
}
const base = new URL(option('--base-url'));
const output = resolve(option('--output-dir'));
if (!base.pathname.endsWith('/')) throw new Error('--base-url must end with /');
if (!['http:', 'https:'].includes(base.protocol)) throw new Error('Serve the existing chart documents over HTTP');
await mkdir(output, {recursive:true});
const profile = await mkdtemp(join(tmpdir(), 'scgsim-chart-preview-'));
const chrome = spawn(process.env.SCGSIM_CHROME || 'google-chrome', [
  '--headless=new', '--no-sandbox', '--disable-dev-shm-usage',
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank'
], {detached:true, stdio:['ignore', 'ignore', 'pipe']});
const exited = new Promise(resolveExit => chrome.once('exit', resolveExit));
let diagnostics = '', launchError;
chrome.stderr.on('data', data => { diagnostics += data; });
chrome.on('error', error => { launchError = error; });
const pause = ms => new Promise(resolvePause => setTimeout(resolvePause, ms));
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
let socket;
try {
  let endpoint;
  for (let i = 0; i < 100; i++) {
    if (launchError) throw launchError;
    endpoint = diagnostics.match(/DevTools listening on ws:\/\/127\.0\.0\.1:(\d+)\//);
    if (endpoint) break;
    if (chrome.exitCode !== null) throw new Error('Chrome exited before its debugging endpoint was ready');
    await pause(100);
  }
  if (!endpoint) throw new Error('Chrome debugging endpoint did not become ready');
  const tabs = await (await fetch(`http://127.0.0.1:${endpoint[1]}/json`)).json();
  socket = new WebSocket(tabs.find(tab => tab.type === 'page').webSocketDebuggerUrl);
  await new Promise((resolveOpen, rejectOpen) => { socket.onopen = resolveOpen; socket.onerror = rejectOpen; });
  let serial = 0;
  const pending = new Map();
  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    if (!message.id) return;
    const waiting = pending.get(message.id);
    if (!waiting) return;
    pending.delete(message.id);
    message.error ? waiting.reject(message.error) : waiting.resolve(message.result);
  };
  const call = (method, params = {}) => new Promise((resolveCall, rejectCall) => {
    const id = ++serial;
    pending.set(id, {resolve:resolveCall, reject:rejectCall});
    socket.send(JSON.stringify({id, method, params}));
  });
  const evaluate = async expression => {
    const result = await call('Runtime.evaluate', {expression, awaitPromise:true, returnByValue:true});
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  await call('Page.enable');
  await call('Emulation.setDeviceMetricsOverride', {width:1000, height:700, deviceScaleFactor:1, mobile:false});
  const figures = [];
  for (const prefix of ['', 'en-']) {
    for (const name of ['source-xy', 'context-xy', 'cross-section', 'planar-arrangement', 'shared-volumes', 'curved-construction-3d', 'construction-3d']) {
      const stem = prefix + name;
      const html = `${stem}.html`;
      const response = await fetch(new URL(html, base));
      if (!response.ok) throw new Error(`Chart request failed: ${html} (${response.status})`);
      const served = Buffer.from(await response.arrayBuffer());
      const local = await readFile(join(output, html));
      if (sha256(served) !== sha256(local)) throw new Error(`Served chart differs from the output source: ${html}`);
      await call('Page.navigate', {url:new URL(html, base).href});
      let ready = false;
      for (let i = 0; i < 150; i++) {
        ready = await evaluate(`document.readyState === 'complete' && !!document.querySelector('.plotly-graph-div')?._fullLayout`);
        if (ready) break;
        await pause(100);
      }
      if (!ready) throw new Error(`Plotly did not finish initializing: ${html}`);
      await pause(200);
      const png = await evaluate(`(async()=>{await document.fonts.ready; const graph=document.querySelector('.plotly-graph-div'); return Plotly.toImage(graph,{format:'png',width:1000,height:700,scale:1.5});})()`);
      if (!png?.startsWith('data:image/png;base64,')) throw new Error(`Invalid image export: ${html}`);
      const bytes = Buffer.from(png.split(',')[1], 'base64');
      const preview = `${stem}-preview.png`;
      await writeFile(join(output, preview), bytes);
      figures.push({chart:html, chart_sha256:sha256(local), preview, preview_sha256:sha256(bytes)});
      console.log(preview);
    }
  }
  await writeFile(join(output, 'chart-previews.json'), JSON.stringify({
    schema:'scgsim.sgb.chart-previews.v1', width:1500, height:1050,
    view:'Default Plotly view; interactive HTML remains authoritative for chart controls.', figures
  }, null, 2) + '\n');
} finally {
  socket?.close();
  if (chrome.pid) {
    try { process.kill(-chrome.pid, 'SIGTERM'); } catch {}
    let timer;
    try {
      await Promise.race([exited, new Promise(resolveTimeout => {timer = setTimeout(() => {
        try { process.kill(-chrome.pid, 'SIGKILL'); } catch {}
        resolveTimeout();
      }, 3000);})]);
      await exited;
    } finally { clearTimeout(timer); }
  }
  await rm(profile, {recursive:true, force:true});
}

const { JSDOM } = require('jsdom');
const fs = require('fs');
const path = require('path');

(async () => {
  const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
  const dom = new JSDOM(html, { runScripts: 'dangerously', resources: 'usable', url: 'http://localhost' });

  await new Promise(resolve => dom.window.document.addEventListener('DOMContentLoaded', resolve));
  await dom.window._transformersReady;
  if (!dom.window.transformers || typeof dom.window.transformers.pipeline !== 'function') {
    throw new Error('transformers pipeline missing');
  }
  dom.window.eval("state.tools.push({name:'echo',desc:'echo',code:'return api.args.msg;'})");
  const res = await dom.window.runTool('echo', { msg: 'hi' });
  console.log('tool ok?', res.ok && res.result === 'hi');
})();

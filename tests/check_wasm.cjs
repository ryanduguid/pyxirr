const { readFileSync } = require('node:fs');
const { basename, join } = require('node:path');
const { loadPyodide } = require('pyodide');

async function main() {
  const wheel = process.argv[2];
  if (!wheel || process.argv.length !== 3) {
    throw new Error('Expected one wheel path');
  }
  const pyodide = await loadPyodide();
  await pyodide.loadPackage(['micropip', 'numpy']);
  const path = `/tmp/${basename(wheel)}`;
  pyodide.FS.writeFile(path, readFileSync(wheel));
  pyodide.globals.set('wheel_path', `emfs://${path}`);
  await pyodide.runPythonAsync(`
import micropip
await micropip.install(wheel_path)
`);
  pyodide.runPython(readFileSync(join(__dirname, 'check_wheel.py'), 'utf8'));
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});

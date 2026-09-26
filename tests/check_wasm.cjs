const { readFileSync } = require('node:fs');
const { basename } = require('node:path');
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
import numpy as np
import pyxirr

dates = ['2020-01-01', '2021-01-01']
assert pyxirr.xnpv(0, dates, [-100, 110]) == 10
assert pyxirr.xnpv(0, dates, np.array([-100.0, 110.0])) == 10
assert pyxirr.xnpv([], dates, [-100, 110]) == []
assert len(pyxirr.xnpv(np.array([]), dates, [-100, 110])) == 0
assert pyxirr.is_conventional_cash_flow([-100, 0, 110])
for rates in (0.1, [], np.array([])):
    try:
        pyxirr.xnpv(rates, [], [])
    except pyxirr.InvalidPaymentsError:
        pass
    else:
        raise AssertionError('Empty payments were accepted')
print('Pyodide wheel smoke passed')
`);
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});

// solver.worker.js — runs IDA* in a background thread
importScripts('cube_core.js');

onmessage = function (e) {
  const state = new Uint8Array(e.data);
  const solution = idaStar(state);
  postMessage(solution);
};

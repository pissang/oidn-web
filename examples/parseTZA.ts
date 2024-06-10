import { parseTZA } from '../src/tza';

fetch('../weights/rt_ldr.tza')
  .then((res) => res.arrayBuffer())
  .then((ab) => {
    const tensors = parseTZA(ab);
    console.log(tensors);
  });

import { initUNetWithModelPath } from '../src/main';

initUNetWithModelPath('../weights/rt_ldr.tza').then((unet) => {
  unet.buildModel();
  console.log(unet);
});

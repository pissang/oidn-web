import * as tfjs from '@tensorflow/tfjs-core';
export function profileAndLogKernelCode(execute: () => void, disabled = true) {
  if (disabled) {
    execute();
    return;
  }
  tfjs
    .profile(() => {
      execute();
    })
    .then((res) => {
      const kernelNames = Array.from(
        new Set(
          res.kernels.map((k) => k.name).filter((name) => !name.endsWith('_op'))
        )
      );
      function nameToConfig(name: string) {
        return `${name[0].toLowerCase()}${name.slice(1)}Config`;
      }
      const importCode = kernelNames.map(
        (name) =>
          `import { ${nameToConfig(
            name
          )} } from '@tensorflow/tfjs-backend-webgpu/dist/kernels/${name}';`
      );
      const configCode = kernelNames.map((name) => `${nameToConfig(name)},`);
      const code = `
  ${importCode.join('\n')}
  const kernelConfigs: KernelConfig[] = [
  ${configCode.join('\n')}
  ]
  `;
      console.log(code);
    });
}

export function memory() {
  return tfjs.memory();
}

export function tidy(f: () => void) {
  return tfjs.tidy(f);
}

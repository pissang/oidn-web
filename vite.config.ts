import { defineConfig } from 'vite';
import mkcert from 'vite-plugin-mkcert';
import * as path from 'path';

export default defineConfig({
  server: {
    host: true,
    hmr: false
  },
  plugins: [mkcert()],

  build: {
    lib: {
      entry: path.resolve(__dirname, 'src/main.ts'),
      name: 'oidn',
      fileName: 'oidn'
    }
  }
});

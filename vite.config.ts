import { defineConfig } from 'vite';
import mkcert from 'vite-plugin-mkcert';

export default defineConfig({
  server: {
    host: true,
    hmr: false
  },
  plugins: [mkcert()]
});

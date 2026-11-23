import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [svelte()],
  base: "/facesift/",
  server: {
    proxy: {
      "/facesift/api": "http://localhost:8000",
      "/facesift/media": "http://localhost:8000",
    },
  },
});

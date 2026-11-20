import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const contractsEntry = fileURLToPath(
  new URL("../../packages/shared-contracts/types/index.ts", import.meta.url),
);
const webDir = fileURLToPath(new URL("./", import.meta.url));
const sharedContractsDir = fileURLToPath(
  new URL("../../packages/shared-contracts", import.meta.url),
);

// Dev server talks to the API directly; in Docker, Nginx does this routing instead.
const apiTarget = process.env.API_PROXY_TARGET ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@contracts": contractsEntry,
    },
  },
  server: {
    port: 5173,
    // The shared contracts live outside this app's root.
    fs: { allow: [webDir, sharedContractsDir] },
    proxy: {
      "/api": { target: apiTarget, changeOrigin: true },
      "/ws": { target: apiTarget, ws: true, changeOrigin: true },
    },
  },
});

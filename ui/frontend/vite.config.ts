import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwind from "@tailwindcss/vite";

// Backend (layer-1 FastAPI) has open CORS, so the SPA calls it directly via VITE_API.
export default defineConfig({
  plugins: [react(), tailwind()],
  server: { host: "127.0.0.1", port: 5273 },
});

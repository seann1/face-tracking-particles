import {defineConfig} from "vite";

export default defineConfig({
    // Ensure built assets use the GitHub Pages repo path.
    // Assumption: this project is deployed to https://<user>.github.io/face-tracking-particles
    // If you deploy to a user/organization site (username.github.io), set base to '/'.
    base: '/face-tracking-particles/',

    // ... other configurations
    server: {
        watch: {
            usePolling: true,
        }
    }
});

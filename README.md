# Nexus UI

Front-end + Colab relay backend for the Nexus Swarm Orchestrator.

## Quick Start

### 1. Front-end (GitHub Pages)

1. Create a GitHub repo (e.g. `nexus-ui`) and commit:
   - `index.html`
   - `README.md`
   - optional: `.gitignore` and `backend/` folder (for reference—Pages only serves static files).
2. Go to **Settings → Pages** and set:
   - **Source:** `main` branch
   - **Folder:** `/ (root)`
3. GitHub will publish the site at  
   `https://<your-username>.github.io/nexus-ui/`  
   (may take a minute to go live).

### 2. Backend (Colab Relay)

1. Open Google Colab and upload `backend/backend.py`.
2. Set a Hugging Face token in the runtime:

   ```python
   import os
   os.environ["HF_TOKEN"] = "hf_XXXXXXXXXXXXXXXXXXXX"
   ```

3. Run the script:

   ```python
   !python backend.py
   ```

4. Wait for the Cloudflare tunnel to print a public URL, e.g.:

   ```
   🔗 Public URL: https://something.trycloudflare.com
   ```

5. Copy that URL.

### 3. Connect Front-end to Backend

1. Visit your GitHub Pages site.
2. Paste the Cloudflare relay URL into **Relay URL** at the top and click **Save**.
3. Hit **Self-Test**. If all checks are green, you’re ready.
4. Use **Send** for single messages or **Launch Mission** for the autonomous loop.

### Notes

- The backend must remain running in Colab; closing the notebook ends the tunnel.
- You can re-run `backend.py` anytime to get a new public URL.
- Only the static front-end is hosted on GitHub Pages; all inference happens through your Colab relay.

Enjoy orchestrating!

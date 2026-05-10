---
description: Walk through setting up a local Pokemon Showdown server for poke-env to connect to.
---

Help the user get a local Pokemon Showdown server running so this project's agents can connect via `poke-env`.

Follow this procedure, adapting to the user's OS (likely Windows — use PowerShell-friendly commands; otherwise bash):

1. **Check prerequisites**: Node.js (>= 18). Run `node --version`. If missing, point them to https://nodejs.org or `winget install OpenJS.NodeJS.LTS`.

2. **Clone Pokemon Showdown** to a sibling directory (not inside this repo):
   ```
   cd ..
   git clone https://github.com/smogon/pokemon-showdown.git
   cd pokemon-showdown
   ```

3. **Install and build**:
   ```
   npm install
   ```
   (No separate build step needed for current versions — `npm install` triggers it.)

4. **Configure for local-only, no-rate-limit play** by copying `config/config-example.js` to `config/config.js` and verifying:
   - `exports.port = 8000;`
   - `exports.noipchecks = true;` (helps for local testing)
   - `exports.usesqlite = false;` if you don't want a user DB

5. **Start the server**:
   ```
   node pokemon-showdown start --no-security
   ```
   `--no-security` disables auth and rate limits — fine for local agent development, **never expose this port publicly**.

6. **Verify** the server is up:
   - Open http://localhost:8000 in a browser — should show the Showdown client.
   - From this repo, a `poke-env` `Player` configured with `server_configuration=LocalhostServerConfiguration` should connect.

7. **Tell the user** how to keep the server running in a separate terminal while they iterate on the agent.

Do not start the server yourself unless the user asks — it's a long-running process. Walk them through it.

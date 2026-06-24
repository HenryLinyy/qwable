# Security Policy

Qwable is designed to bind to `127.0.0.1`. It has no authentication layer and must not be exposed directly to an untrusted network.

## Reporting a vulnerability

Please use GitHub's private security advisory feature for this repository. Do not include credentials, private prompts, or exploitable details in a public issue.

Include the affected version, reproduction steps, and expected impact. Security fixes target the latest release.

## Local data

- Keep `.env` private; it is ignored by Git.
- Conversations are stored under `~/.qwable/conversations` when persistence is used.
- Review model and prompt data before connecting any non-local backend.

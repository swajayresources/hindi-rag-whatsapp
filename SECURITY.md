# Security Policy

## Reporting a vulnerability

Open a private security advisory on GitHub, or email the maintainer. Please do
not file a public issue for anything exploitable.

## Defaults this service relies on

| Control | Default | Why |
|---|---|---|
| Twilio signature validation | **enabled** | Prevents forged webhook calls (denial-of-wallet / prompt injection) |
| Per-client rate limiting | 30 req / 60 s | Caps abuse and LLM spend |
| Question length cap | 1,000 chars | Bounds prompt size and cost |
| LLM credentials required in production | enforced | Prevents silent misconfiguration |
| Structured errors | no stack traces | Avoids leaking internals |
| Secrets | `SecretStr`, env only | Never logged; excluded from repr |

## If you disable a control

`TWILIO_VALIDATE_SIGNATURE=false` is intended **only** for local webhook testing.
Never ship it to a public URL.

## Secret handling

- `.env` is git-ignored.
- The Docker image contains no secrets; inject them at runtime.
- Rotate `GROQ_API_KEY` and Twilio credentials if a build log ever contains them.

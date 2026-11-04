# OpenCode V2 connection

1. Keep Strata and Cloudflare Tunnel running in Kaggle.
2. Copy the temporary HTTPS URL from the notebook output.
3. In the Kaggle shell, run `strata launch --show-key` and copy the key privately.
4. Save the key in a local ignored file: `~/.config/opencode/strata-key.txt`.
5. Merge the provider object printed by the notebook into the existing `providers` map in `opencode.jsonc`. Do not replace the rest of the config.
6. Restart OpenCode and select the `strata/<model-id>` model.
7. Start a fresh OpenCode session after changing provider or context settings.

Keep the key out of GitHub and chat. The Quick Tunnel URL is temporary and changes after tunnel restart.

8. The generated model entry sets `compatibility.reasoningField` to
   `reasoning_content`, so Strata's separate reasoning channel is treated as
   thinking instead of answer text. It also sets `headerTimeout` and
   `chunkTimeout` to 15 minutes: a T4 can spend minutes on the first token and
   can pause mid-answer while the engine pages experts.

The generated `limit.context` comes from the notebook's `CONTEXT` setting. If you
change `CONTEXT`, regenerate the entry and start a fresh OpenCode session.

### Optional: publish the key encrypted

Set `CLIENT_PUBLIC_KEY` in the configuration cell to a PEM RSA public key and the
notebook prints `STRATA_CONNECTION={"base_url": ..., "encrypted_api_key": ...}`.
Only the holder of the private key can decrypt it, so the API key never appears
in the notebook output and does not need to be stored as a Kaggle Secret. Leave
it blank to keep the manual `strata launch --show-key` flow above.

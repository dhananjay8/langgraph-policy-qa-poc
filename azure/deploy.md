# Azure deployment record

Subscription: `QFlow1-Dev-Elephant-dsp-S` (0b04c716-…), tenant 999fb958-…
Region: eastus. All resources in `rg-langgraph-eval-poc`.

## Resources

```bash
az group create --name rg-langgraph-eval-poc --location eastus

az cognitiveservices account create \
  --name aoai-lgpoc-eval -g rg-langgraph-eval-poc \
  --location eastus --kind OpenAI --sku S0

az cognitiveservices account deployment create \
  --name aoai-lgpoc-eval -g rg-langgraph-eval-poc \
  --deployment-name gpt-5-nano \
  --model-name gpt-5-nano --model-version 2025-08-07 \
  --model-format OpenAI --sku-name GlobalStandard --sku-capacity 30

az acr create --name acrlgpoc2675231837 -g rg-langgraph-eval-poc \
  --location eastus --sku Basic --admin-enabled false

az acr build --registry acrlgpoc2675231837 --image policy-qa:0.1.0 .

az containerapp env create --name cae-lgpoc -g rg-langgraph-eval-poc --location eastus

az containerapp create \
  --name ca-policy-qa -g rg-langgraph-eval-poc --environment cae-lgpoc \
  --image acrlgpoc2675231837.azurecr.io/policy-qa:0.1.0 \
  --registry-server acrlgpoc2675231837.azurecr.io \
  --registry-identity system --system-assigned \
  --target-port 8000 --ingress external \
  --min-replicas 1 --max-replicas 1 --cpu 0.5 --memory 1Gi \
  --secrets poc-api-key=<redacted> aoai-key=<redacted> \
  --env-vars POC_API_KEY=secretref:poc-api-key \
             AZURE_OPENAI_API_KEY=secretref:aoai-key \
             AZURE_OPENAI_ENDPOINT=<endpoint> \
             AZURE_OPENAI_DEPLOYMENT=gpt-5-nano \
             AZURE_OPENAI_API_VERSION=2024-10-21
```

Endpoint: `https://ca-policy-qa.greenpond-2080be32.eastus.azurecontainerapps.io`

The CLI automatically granted the app's system-assigned identity `AcrPull`
on the registry.

## Notes

- `gpt-4o-mini` could not be deployed (model in deprecating state);
  `gpt-5-nano` was used instead.
- The container image was built remotely with `az acr build` because no
  local Docker daemon is available on this machine.
- Auth on the API is an `X-API-Key` header checked with constant-time
  comparison; `/healthz` is unauthenticated.

## Teardown

```bash
az group delete --name rg-langgraph-eval-poc --yes --no-wait
```

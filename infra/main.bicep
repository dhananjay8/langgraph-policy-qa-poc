// ---------------------------------------------------------------------------
// LangGraph Policy Q&A POC — Azure infrastructure (Bicep)
//
// Deploys: Resource Group scoped resources
//   - Azure OpenAI (S0) with gpt-5-nano deployment
//   - Container Registry (Basic)
//   - Log Analytics workspace + Application Insights
//   - Container Apps environment + container app
//
// Usage:
//   az deployment group create \
//     --resource-group rg-langgraph-eval-poc \
//     --template-file infra/main.bicep \
//     --parameters infra/parameters.json
// ---------------------------------------------------------------------------

@description('Azure region for all resources')
param location string = resourceGroup().location

@description('Unique suffix for globally unique names (default: first 13 chars of resource group id)')
param uniqueSuffix string = uniqueString(resourceGroup().id)

@description('Azure OpenAI model deployment name')
param openaiDeploymentName string = 'gpt-5-nano'

@description('Azure OpenAI model name')
param openaiModelName string = 'gpt-5-nano'

@description('Azure OpenAI model version')
param openaiModelVersion string = '2025-08-07'

@description('Azure OpenAI API version')
param openaiApiVersion string = '2024-10-21'

@description('Container image tag')
param imageTag string = '0.2.0'

@secure()
@description('API key for the POC service')
param pocApiKey string

@secure()
@description('Azure OpenAI API key (auto-resolved if empty)')
param openaiApiKey string = ''

// ---------------------------------------------------------------------------
// Azure OpenAI
// ---------------------------------------------------------------------------

resource openai 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: 'aoai-lgpoc-${uniqueSuffix}'
  location: location
  kind: 'OpenAI'
  sku: { name: 'S0' }
  properties: {
    customSubDomainName: 'aoai-lgpoc-${uniqueSuffix}'
    publicNetworkAccess: 'Enabled'
  }
}

resource openaiDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openai
  name: openaiDeploymentName
  sku: {
    name: 'GlobalStandard'
    capacity: 30
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: openaiModelName
      version: openaiModelVersion
    }
  }
}

// ---------------------------------------------------------------------------
// Container Registry
// ---------------------------------------------------------------------------

resource acr 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' = {
  name: 'acrlgpoc${uniqueSuffix}'
  location: location
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: false }
}

// ---------------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------------

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'law-lgpoc-${uniqueSuffix}'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'ai-policy-qa-${uniqueSuffix}'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
  }
}

// ---------------------------------------------------------------------------
// Container Apps
// ---------------------------------------------------------------------------

resource containerEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'cae-lgpoc-${uniqueSuffix}'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

var resolvedOpenaiKey = empty(openaiApiKey) ? openai.listKeys().key1 : openaiApiKey

resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'ca-policy-qa'
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    managedEnvironmentId: containerEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'http'
      }
      registries: [
        {
          server: acr.properties.loginServer
          identity: 'system'
        }
      ]
      secrets: [
        { name: 'poc-api-key', value: pocApiKey }
        { name: 'aoai-key', value: resolvedOpenaiKey }
        { name: 'appinsights-conn', value: appInsights.properties.ConnectionString }
      ]
    }
    template: {
      containers: [
        {
          name: 'policy-qa'
          image: '${acr.properties.loginServer}/policy-qa:${imageTag}'
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            { name: 'POC_API_KEY', secretRef: 'poc-api-key' }
            { name: 'AZURE_OPENAI_API_KEY', secretRef: 'aoai-key' }
            { name: 'AZURE_OPENAI_ENDPOINT', value: openai.properties.endpoint }
            { name: 'AZURE_OPENAI_DEPLOYMENT', value: openaiDeploymentName }
            { name: 'AZURE_OPENAI_API_VERSION', value: openaiApiVersion }
            { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', secretRef: 'appinsights-conn' }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 3
        rules: [
          {
            name: 'http-scaling'
            http: { metadata: { concurrentRequests: '20' } }
          }
        ]
      }
    }
  }
}

// Grant AcrPull to the container app's system identity
resource acrPullRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, containerApp.id, 'acrpull')
  scope: acr
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
    principalId: containerApp.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------

output appUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
output acrLoginServer string = acr.properties.loginServer
output openaiEndpoint string = openai.properties.endpoint
// NOTE: appInsightsConnectionString intentionally omitted from outputs
// (contains ingestion key). Retrieve via: az monitor app-insights component show ...

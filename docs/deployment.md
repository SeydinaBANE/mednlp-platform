# MedNLP Platform — Guide de déploiement

## Vue d'ensemble

Le déploiement repose sur GitHub Actions et Google Cloud Platform (GCP). Deux environnements existent :

| Environnement | Déclencheur | Cible |
|---------------|-------------|-------|
| **Staging** | Merge automatique sur `main` | Cloud Run `mednlp-api-staging` |
| **Production** | `workflow_dispatch` manuel + approbation | Cloud Run `mednlp-api-prod` |

---

## Prérequis

- Compte GCP avec facturation activée
- CLI `gcloud` installé ([installation](https://cloud.google.com/sdk/docs/install))
- Accès Owner ou Editor au projet GCP
- Accès Admin au repo GitHub

---

## 1. Initialisation GCP (une seule fois)

### 1.1 Créer le projet

```bash
gcloud projects create mednlp-platform --name "MedNLP Platform"
gcloud config set project mednlp-platform
```

> Note le **Project ID** (ex: `mednlp-platform-123456`) — il sera nécessaire partout.

### 1.2 Activer les APIs

```bash
gcloud services enable \
  run.googleapis.com \
  containerregistry.googleapis.com \
  secretmanager.googleapis.com \
  pubsub.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  aiplatform.googleapis.com
```

### 1.3 Créer le Service Account de déploiement

```bash
export PROJECT_ID=$(gcloud config get-value project)

gcloud iam service-accounts create deploy-sa \
  --display-name "GitHub Actions Deploy"

# Droits Cloud Run
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member "serviceAccount:deploy-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role "roles/run.admin"

# Droits Container Registry (push d'images Docker)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member "serviceAccount:deploy-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role "roles/storage.admin"

# Droits Secret Manager (lecture des secrets applicatifs)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member "serviceAccount:deploy-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role "roles/secretmanager.secretAccessor"

# Droits pour agir en tant que service account (Cloud Run Jobs)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member "serviceAccount:deploy-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role "roles/iam.serviceAccountUser"
```

### 1.4 Configurer Workload Identity Federation

Permet à GitHub Actions de s'authentifier à GCP sans stocker de clé JSON.

```bash
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")

# Créer le pool
gcloud iam workload-identity-pools create github-pool \
  --location global \
  --display-name "GitHub Actions Pool"

# Créer le provider OIDC
gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location global \
  --workload-identity-pool github-pool \
  --issuer-uri "https://token.actions.githubusercontent.com" \
  --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition "assertion.repository=='SeydinaBANE/mednlp-platform'"

# Autoriser le repo GitHub à impersonner le service account
gcloud iam service-accounts add-iam-policy-binding \
  deploy-sa@$PROJECT_ID.iam.gserviceaccount.com \
  --role "roles/iam.workloadIdentityUser" \
  --member "principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/SeydinaBANE/mednlp-platform"
```

Récupérer la valeur du provider pour GitHub Secrets :

```bash
gcloud iam workload-identity-pools providers describe github-provider \
  --location global \
  --workload-identity-pool github-pool \
  --format "value(name)"
# → projects/123456789/locations/global/workloadIdentityPools/github-pool/providers/github-provider
```

### 1.5 Créer les secrets applicatifs

```bash
# Clé OpenRouter (LLM)
echo -n "sk-or-v1-..." | gcloud secrets create openrouter-api-key --data-file=-

# Clé secrète de l'application (JWT)
openssl rand -hex 32 | gcloud secrets create app-secret-key --data-file=-

# Clé production (si différente du staging)
openssl rand -hex 32 | gcloud secrets create app-secret-key-prod --data-file=-

# Webhook Slack (alertes deploy)
echo -n "https://hooks.slack.com/..." | gcloud secrets create slack-webhook --data-file=-
```

---

## 2. Configuration GitHub

### 2.1 Secrets (Settings → Secrets and variables → Actions)

| Nom | Description | Exemple |
|-----|-------------|---------|
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Output de l'étape 1.4 | `projects/123.../providers/github-provider` |
| `GCP_SERVICE_ACCOUNT` | Email du service account | `deploy-sa@mednlp-platform.iam.gserviceaccount.com` |
| `SLACK_WEBHOOK_URL` | Webhook pour notifications deploy | `https://hooks.slack.com/services/...` |

### 2.2 Variables (Settings → Secrets and variables → Actions → Variables)

| Nom | Description | Exemple |
|-----|-------------|---------|
| `GCP_PROJECT_ID` | ID du projet GCP | `mednlp-platform-123456` |
| `GCP_REGION` | Région de déploiement | `europe-west1` |

---

## 3. Déploiement staging (automatique)

Le CD staging se déclenche automatiquement à chaque merge sur `main`.

```
push → main
  └─ CI (lint + tests + Docker scan)
  └─ CD Staging
       ├─ Build image Docker → push vers GCR
       ├─ Deploy Cloud Run (mednlp-api-staging) sans trafic
       ├─ Run migration Alembic
       ├─ Bascule 100% du trafic vers la nouvelle révision
       ├─ Health check /health
       └─ Notification Slack ✅ ou ❌
```

Pour déclencher manuellement sans merger :

```bash
gh workflow run cd-staging.yml
# ou via l'UI : Actions → CD — Staging → Run workflow
```

---

## 4. Déploiement production (manuel)

Le déploiement production est **entièrement manuel** et nécessite une approbation dans GitHub Environments.

```bash
# Via CLI
gh workflow run cd-production.yml \
  -f confirm=deploy \
  -f canary_weight=5

# Via l'UI : Actions → CD — Production → Run workflow
#   confirm: deploy
#   canary_weight: 5  (5% du trafic envoyé vers la nouvelle version)
```

Séquence :
1. Gate check : relance les tests unitaires
2. Approbation manuelle dans GitHub (environment `production`)
3. Build & push image vers GCR
4. Migration Alembic
5. Deploy sans trafic
6. Bascule canary (5% par défaut, configurable)
7. Monitoring 5 minutes + health check
8. Bascule 100% si OK, rollback si KO
9. Notification Slack

### Rollback production

```bash
# Lister les révisions Cloud Run
gcloud run revisions list --service mednlp-api-prod --region REGION

# Basculer vers une révision précédente
gcloud run services update-traffic mednlp-api-prod \
  --region REGION \
  --to-revisions mednlp-api-prod-00042-xyz=100
```

---

## 5. Fine-tuning et backfill (opérations ponctuelles)

### Fine-tuning sur Vertex AI

```bash
make fine-tune TASK=icd10    # entraînement classifieur ICD-10
make fine-tune TASK=triage   # entraînement classifieur triage
```

Lance un job Vertex AI sur `n1-standard-8 + T4`. Le modèle promu passe au stage `Production` dans MLflow.

### Backfill de vectorisation

```bash
make backfill MODEL=v2   # re-vectorise tout le corpus avec le modèle v2
```

À lancer après la promotion d'un nouveau modèle d'embedding en production.

---

## 6. Vérifications post-déploiement

```bash
# Health check API
curl https://mednlp-api-staging-<hash>.run.app/health

# Logs en temps réel
gcloud run services logs tail mednlp-api-staging --region REGION

# Métriques Cloud Run
gcloud run services describe mednlp-api-staging --region REGION
```

Les métriques Prometheus sont exposées sur le port 9090 en production et consultables via Grafana Cloud Monitoring.

---

## 7. Variables d'environnement de référence

Voir `.env.example` à la racine du projet pour la liste complète des variables. En production, toutes les valeurs sensibles sont injectées via **Secret Manager** — aucune valeur en clair dans les workflows.

# ==============================================================================
# PramanAI: 1-Click Production Deployment to Google Cloud Run (asia-south1)
# Track 3: The Fortified Enterprise Fleet on Google Cloud & Gemini 3.5 Flash
# ==============================================================================

param (
    [string]$ProjectId = $env:GOOGLE_CLOUD_PROJECT,
    [string]$Region = "asia-south1",
    [string]$ServiceName = "praman-ai-backend",
    [string]$GcsBucketName = $env:GCS_BUCKET_NAME
)

if (-not $ProjectId) {
    $ProjectId = "praman-ai-govtech"
}
if (-not $GcsBucketName) {
    $GcsBucketName = "praman-ai-documents-$ProjectId"
}

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " PramanAI Cloud Run Deployment (Google All Things Agentic)" -ForegroundColor Cyan
Write-Host " Track 3: Fortified Enterprise Fleet (Gemini 3.5 Flash & GCP)" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " Project ID   : $ProjectId" -ForegroundColor Yellow
Write-Host " Target Region: $Region" -ForegroundColor Yellow
Write-Host " Cloud Service: $ServiceName" -ForegroundColor Yellow
Write-Host " GCS Bucket   : $GcsBucketName" -ForegroundColor Yellow
Write-Host ""

# 1. Verify gcloud CLI
Write-Host "[1/6] Verifying Google Cloud SDK installation..." -ForegroundColor Green
$gcloudCmd = Get-Command gcloud -ErrorAction SilentlyContinue
if (-not $gcloudCmd) {
    Write-Error "Google Cloud SDK (gcloud) is not installed or not in PATH. Please install Google Cloud SDK."
    exit 1
}

# 2. Configure Google Cloud Project
Write-Host "[2/6] Configuring Google Cloud active project ($ProjectId)..." -ForegroundColor Green
gcloud config set project $ProjectId --quiet

# 3. Enable Required Google Cloud APIs
Write-Host "[3/6] Enabling Google Cloud Services (Cloud Run, Cloud Build, Artifact Registry, Storage, Vertex AI)..." -ForegroundColor Green
gcloud services enable `
    run.googleapis.com `
    cloudbuild.googleapis.com `
    artifactregistry.googleapis.com `
    storage.googleapis.com `
    aiplatform.googleapis.com `
    --quiet

# 4. Create Google Cloud Storage Bucket for Evidentiary Documents
Write-Host "[4/6] Verifying Google Cloud Storage document bucket (gs://$GcsBucketName)..." -ForegroundColor Green
$bucketExists = gsutil ls -b "gs://$GcsBucketName" 2>$null
if (-not $bucketExists) {
    Write-Host "Creating GCS Bucket: gs://$GcsBucketName in $Region..." -ForegroundColor Yellow
    gsutil mb -l $Region "gs://$GcsBucketName"
} else {
    Write-Host "GCS Bucket gs://$GcsBucketName already exists." -ForegroundColor Gray
}

# 5. Build and Deploy Container to Google Cloud Run
Write-Host "[5/6] Deploying PramanAI FastAPI Backend to Google Cloud Run ($Region)..." -ForegroundColor Green
$envVars = @(
    "GEMINI_FLASH_MODEL=gemini-3.5-flash",
    "GEMINI_LITE_MODEL=gemini-3.5-flash-lite",
    "GEMINI_ARMOR_MODEL=gemma-2-2b-it",
    "GOOGLE_CLOUD_PROJECT=$ProjectId",
    "GCS_BUCKET_NAME=$GcsBucketName",
    "GOOGLE_CLOUD_REGION=$Region"
)

if ($env:GEMINI_API_KEY) {
    $envVars += "GEMINI_API_KEY=$($env:GEMINI_API_KEY)"
}
if ($env:DATABASE_URL) {
    $envVars += "DATABASE_URL=$($env:DATABASE_URL)"
}

$envString = $envVars -join ","

gcloud run deploy $ServiceName `
    --source . `
    --region $Region `
    --platform managed `
    --allow-unauthenticated `
    --port 8000 `
    --memory 2Gi `
    --cpu 2 `
    --min-instances 0 `
    --max-instances 10 `
    --concurrency 80 `
    --set-env-vars "$envString" `
    --quiet

if ($LASTEXITCODE -ne 0) {
    Write-Error "Cloud Run deployment failed. Check build logs above."
    exit 1
}

# 6. Retrieve Service URL and Verify Health
$serviceUrl = gcloud run services describe $ServiceName --region $Region --format "value(status.url)"

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " [SUCCESS] PramanAI Deployed to Google Cloud Run Live!" -ForegroundColor Green
Write-Host " Service URL    : $serviceUrl" -ForegroundColor White
Write-Host " Health Check   : $serviceUrl/health" -ForegroundColor Yellow
Write-Host " API Docs (OpenAPI): $serviceUrl/docs" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Cyan

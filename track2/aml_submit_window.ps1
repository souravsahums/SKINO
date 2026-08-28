# Open the workspace storage to THIS machine's IP just long enough to upload the
# code snapshot and create the jobs, then guarantee it is closed again.
#
# The account has publicNetworkAccess=Disabled by default. Compute nodes reach it
# over the private endpoint; only the client-side code upload needs the public
# endpoint. We open it in a TIGHTER posture than the default (deny-all except our
# single IP) and restore the exact original posture in a finally block so the
# firewall can never be left open even if submission throws.
param(
    [string[]]$Stages = @("1d", "2d", "3d"),
    [int]$Shards = 2
)
$ErrorActionPreference = "Continue"   # az writes progress to stderr; do not treat that as terminating
$Rg = $env:AML_RESOURCE_GROUP; $Sa = $env:AML_STORAGE_ACCOUNT
$said = "/subscriptions/$($env:AML_SUBSCRIPTION_ID)/resourceGroups/$Rg/providers/Microsoft.Storage/storageAccounts/$Sa"
$py = $env:PYTHON_EXE
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$myip = (Invoke-RestMethod -Uri "https://api.ipify.org?format=json" -TimeoutSec 15).ip
$origPna = az storage account show --ids $said --query publicNetworkAccess -o tsv
$origDefault = az storage account show --ids $said --query networkRuleSet.defaultAction -o tsv
Write-Host "IP=$myip  ORIGINAL publicNetworkAccess=$origPna defaultAction=$origDefault"

$submitted = $false
try {
    az storage account update --ids $said --public-network-access Enabled --default-action Deny -o none
    if ($LASTEXITCODE -ne 0) { throw "failed to enable public network access" }
    az storage account network-rule add -g $Rg --account-name $Sa --ip-address $myip -o none
    if ($LASTEXITCODE -ne 0) { throw "failed to add IP allow rule" }

    # Readiness probe: storage ACL changes take a little while to take effect, so
    # poll the data plane (not a job) until our IP is accepted before submitting.
    $ready = $false; $lastErr = ""
    for ($i = 1; $i -le 12; $i++) {
        $lastErr = (az storage container list --account-name $Sa --auth-mode login --num-results 1 2>&1 | Out-String)
        if ($LASTEXITCODE -eq 0) { $ready = $true; break }
        Write-Host "storage not reachable yet (attempt $i/12); waiting for ACL propagation"
        Start-Sleep -Seconds 20
    }
    if (-not $ready) { throw "storage not reachable after opening firewall. last error: $lastErr" }
    Write-Host "storage reachable; submitting stages: $($Stages -join ', ')"

    foreach ($stage in $Stages) {
        Write-Host "== submit stage $stage (shards=$Shards) =="
        & $py -m track2.azureml_submit --stage $stage --shards $Shards
        if ($LASTEXITCODE -ne 0) { throw "submit failed for stage $stage" }
    }
    $submitted = $true
}
finally {
    az storage account network-rule remove -g $Rg --account-name $Sa --ip-address $myip -o none 2>$null
    az storage account update --ids $said --public-network-access $origPna --default-action $origDefault -o none
    $chk = az storage account show --ids $said --query "{pna:publicNetworkAccess,def:networkRuleSet.defaultAction}" -o json
    Write-Host "FIREWALL RESTORED -> $chk   submitted=$submitted"
}

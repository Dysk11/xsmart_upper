param(
    [string]$HostName = "192.168.32.84",
    [string]$User = "orangepi",
    [string]$IdentityFile = "C:\Users\Dysk11\.ssh\id_ed25519_orangepi",
    [string]$RemoteProject = "/home/orangepi/Downloads/xsmart_upper/xsmart_upper",
    [string]$Video = "outputs/video/record_20260708_135111.mp4",
    [switch]$Scout,
    [string[]]$Only = @()
)

$ErrorActionPreference = "Stop"
$Target = "$User@$HostName"
$CommonSsh = @(
    "-o", "BatchMode=yes",
    "-o", "PasswordAuthentication=no",
    "-o", "IdentitiesOnly=yes",
    "-o", "StrictHostKeyChecking=no",
    "-i", $IdentityFile
)
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RemoteScript = "/tmp/xsmart_benchmark_$Stamp.py"
$RemoteOutput = "$RemoteProject/outputs/benchmark/rknn_benchmark_$Stamp.json"

try {
    & scp @CommonSsh "$PSScriptRoot\benchmark_rknn.py" "${Target}:$RemoteScript"
    if ($LASTEXITCODE -ne 0) { throw "Failed to upload benchmark script" }

    $Arguments = @(
        "XSMART_PROJECT_ROOT='$RemoteProject'",
        "python3", "'$RemoteScript'",
        "--video", "'$RemoteProject/$Video'",
        "--output", "'$RemoteOutput'"
    )
    if ($Scout) { $Arguments += "--scout" }
    foreach ($Name in $Only) {
        $Arguments += @("--only", "'$Name'")
    }
    & ssh @CommonSsh $Target ($Arguments -join " ")
    if ($LASTEXITCODE -ne 0) { throw "Remote benchmark failed" }

    New-Item -ItemType Directory -Force "$PSScriptRoot\..\outputs\benchmark" | Out-Null
    & scp @CommonSsh "${Target}:$RemoteOutput" "$PSScriptRoot\..\outputs\benchmark\"
    & scp @CommonSsh "${Target}:$($RemoteOutput -replace '\.json$', '.csv')" "$PSScriptRoot\..\outputs\benchmark\"
    if ($LASTEXITCODE -ne 0) { throw "Failed to download benchmark results" }
}
finally {
    & ssh @CommonSsh $Target "rm -f '$RemoteScript'" | Out-Null
}

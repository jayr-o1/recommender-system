# PowerShell script to start the API server using Uvicorn

# Set the working directory to the script's location
Set-Location -Path $PSScriptRoot

# Define variables
$Host = "0.0.0.0"  # Listen on all interfaces
$Port = 8000
$Module = "src.api:app"
$ReloadFlag = $true  # Enable auto-reload for development

# Show startup message
Write-Host "Starting Career Recommender API with Uvicorn..."
Write-Host "Host: $Host"
Write-Host "Port: $Port"
Write-Host "Press Ctrl+C to stop the server."

# Start the API server with Uvicorn
if ($ReloadFlag) {
    # Development mode with auto-reload
    uvicorn $Module --host $Host --port $Port --reload
} else {
    # Production mode without auto-reload
    uvicorn $Module --host $Host --port $Port
} 
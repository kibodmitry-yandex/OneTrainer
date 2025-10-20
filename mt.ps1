Push-Location -LiteralPath $PSScriptRoot
try {
	# Activate virtualenv (run the activation script in current shell)
	& "./venv/Scripts/Activate.ps1"
	# Run MaskingTool from repository root so package imports resolve
	& "./venv/Scripts/python.exe" "./modules/ui/MaskingTool.py"
} finally {
	Pop-Location
}
<#
setup-latex-workshop-windows.ps1

Downloads the TeX Live Windows installer to a safe folder and opens
the Ghostscript download page. This script DOES NOT auto-run installers.

Usage: Run in PowerShell (no elevated rights required to download files):
  powershell -ExecutionPolicy Bypass -File .\setup-latex-workshop-windows.ps1

After downloading, run the installers manually and then restart Windows
and VS Code.
#>

$OutDir = Join-Path $env:USERPROFILE 'Downloads\LaTeX-Setup'
New-Item -Path $OutDir -ItemType Directory -Force | Out-Null

Write-Output "Downloading to: $OutDir"

# TeX Live installer (CTAN mirror redirect)
$texUri = 'http://mirror.ctan.org/systems/texlive/tlnet/install-tl-windows.exe'
$texOut = Join-Path $OutDir 'install-tl-windows.exe'
Write-Output "Downloading TeX Live installer..."
Invoke-WebRequest -Uri $texUri -OutFile $texOut -UseBasicParsing -ErrorAction Stop
Write-Output "Saved TeX Live installer to: $texOut"

# Open Ghostscript download page (user will manually download installer)
$gsPage = 'https://www.ghostscript.com/download/gsdnld.html'
Write-Output "Opening Ghostscript download page in default browser..."
Start-Process $gsPage

Write-Output ""
Write-Output "Next steps (manual):"
Write-Output "- Run the TeX Live installer: $texOut"
Write-Output "- Download and run Ghostscript from the opened page."
Write-Output "- If you chose MiKTeX instead of TeX Live, install Strawberry Perl (https://strawberryperl.com/) so latexmk works."
Write-Output "- (Optional) Install ImageMagick and chktex."

Write-Output ""
Write-Output "Suggested PATH example for TeX Live (adjust year):"
Write-Output "  C:\\texlive\\2025\\bin\\win32"
Write-Output "If the TeX Live installer added the bin to PATH, restart Windows and VS Code."

Write-Output ""
Write-Output "To verify after installation, run these commands in PowerShell:"
Write-Output "  where.exe latexmk"
Write-Output "  where.exe xelatex"
Write-Output "  where.exe kpsewhich"
Write-Output "  where.exe biber"
Write-Output "  where.exe gswin64c"
Write-Output "  magick -version    # if ImageMagick installed"

Write-Output ""
Write-Output "If you want, I can also provide a script to install optional tools via Chocolatey (requires admin)."

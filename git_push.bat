@echo off
title JumpToThis GitHub Uploader
echo ===================================================
echo       JumpToThis GitHub Repository Uploader       
echo ===================================================
echo.

echo [*] Resetting Git history to remove large files...
if exist .git (
    rd /s /q .git
)

:: Configure git ignore to exclude virtual environments and raw full video files
echo .venv/ > .gitignore
echo __pycache__/ >> .gitignore
echo scratch/ >> .gitignore
echo media/*_full* >> .gitignore
echo media/cloudflared.exe >> .gitignore
echo .idea/ >> .gitignore
echo .vscode/ >> .gitignore

echo [*] Re-initializing Git...
git init
git add .
git commit -m "Clean project commit excluding large raw video assets"
echo.

set /p REPO_URL="👉 Paste your GitHub Repository URL (e.g. https://github.com/user/repo) and press Enter: "
echo.

echo [*] Connecting to remote repository...
git remote add origin %REPO_URL%
git branch -M main

echo [*] Pushing clean repository to GitHub...
git push -u origin main -f

echo.
echo [+] Code pushed successfully! You can now open this repository on GitHub and launch your Codespace!
pause

@echo off
title JumpToThis Local Launcher
echo ===================================================
echo       JumpToThis Local Server Launcher            
echo ===================================================
echo.
echo [*] Starting FastAPI Backend on port 8000...
start "JumpToThis Server Backend" cmd /k ".venv\Scripts\python backend/app.py"

echo [*] Loading heavy AI models into memory...
powershell -Command "Write-Host -NoNewline '[*] Loading'; while ($true) { try { $c = New-Object System.Net.Sockets.TcpClient('127.0.0.1', 8000); $c.Dispose(); break } catch { Write-Host -NoNewline '.'; Start-Sleep -Seconds 1 } }"
echo.
echo [*] Server is ready! Opening prototype in browser...
start http://127.0.0.1:8000/
echo.
echo [+] Active on http://127.0.0.1:8000
echo [+] You can close this presenter window now.
echo.
pause

@echo off
setlocal EnableExtensions

rem LUMEN-PS local launcher.  This project deliberately has no Node/npm step.
set "ROOT=%~dp0"
cd /d "%ROOT%"
set "PORT=8756"
set "URL=http://127.0.0.1:%PORT%"
set "VENV=%ROOT%.venv"
set "VENV_PY=%VENV%\Scripts\python.exe"
set "RUNTIME=%ROOT%.lumen-ps"
set "LUMEN_ICON=%RUNTIME%\lumen-ps.ico"

title LUMEN-PS - Photometric Stereo Bench
echo.
echo  ================================================
echo    LUMEN-PS  ^|  PHOTOMETRIC STEREO BENCH
echo  ================================================
echo.

rem Prefer the Windows Python launcher so the requested Python 3 interpreter is used.
where py >nul 2>nul
if not errorlevel 1 (
  py -3 -c "import sys" >nul 2>nul
  if not errorlevel 1 set "BOOTSTRAP_PY=py -3"
)
if not defined BOOTSTRAP_PY (
  where python >nul 2>nul
  if errorlevel 1 goto :no_python
  python -c "import sys; raise SystemExit(not (sys.version_info >= (3, 11)))" >nul 2>nul
  if errorlevel 1 goto :no_python
  set "BOOTSTRAP_PY=python"
)

if not exist "%VENV_PY%" (
  echo [setup] Creating local Python environment...
  call %BOOTSTRAP_PY% -m venv "%VENV%"
  if errorlevel 1 goto :setup_failed
)

echo [setup] Checking required Python packages...
"%VENV_PY%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :setup_failed

rem Make the icon and user shortcuts once. Failures here do not stop the bench.
if not exist "%RUNTIME%" mkdir "%RUNTIME%" >nul 2>nul
if not exist "%LUMEN_ICON%" call :make_icon
call :make_shortcuts

rem Do not start a second server if this launcher is opened again.
netstat -ano | findstr /r /c:":%PORT% .*LISTENING" >nul 2>nul
if not errorlevel 1 (
  echo [bench] LUMEN-PS is already running at %URL%
  start "" "%URL%"
  goto :end
)

echo.
echo [bench] Starting local server at %URL%
echo [bench] The scanner status will appear on the boot screen.
echo [bench] Keep this window open while using LUMEN-PS. Press Ctrl+C to stop it.
echo.

rem Wait briefly so the browser receives the boot screen rather than a connection error.
start "" /b powershell.exe -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process '%URL%'"
"%VENV_PY%" -m uvicorn leafscan.web.app:app --host 127.0.0.1 --port %PORT%
goto :end

:make_icon
rem A small amber leaf icon for the Desktop and Start-menu shortcuts.
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Add-Type -AssemblyName System.Drawing; $bmp=[Drawing.Bitmap]::new(64,64); $g=[Drawing.Graphics]::FromImage($bmp); $g.SmoothingMode='AntiAlias'; $g.Clear([Drawing.Color]::Transparent); $leaf=[Drawing.Drawing2D.GraphicsPath]::new(); $leaf.AddBezier([Drawing.PointF]::new(32,7),[Drawing.PointF]::new(53,11),[Drawing.PointF]::new(58,30),[Drawing.PointF]::new(32,52)); $leaf.AddBezier([Drawing.PointF]::new(32,52),[Drawing.PointF]::new(7,42),[Drawing.PointF]::new(9,16),[Drawing.PointF]::new(32,7)); $g.FillPath([Drawing.SolidBrush]::new([Drawing.Color]::FromArgb(255,180,84)),$leaf); $pen=[Drawing.Pen]::new([Drawing.Color]::FromArgb(82,47,15),3); $g.DrawLine($pen,32,14,32,58); $g.DrawLine($pen,32,35,18,25); $g.DrawLine($pen,32,41,45,27); $ico=[Drawing.Icon]::FromHandle($bmp.GetHicon()); $stream=[IO.File]::Open($env:LUMEN_ICON,[IO.FileMode]::Create); $ico.Save($stream); $stream.Close(); $ico.Dispose(); $pen.Dispose(); $g.Dispose(); $bmp.Dispose() } catch {}" >nul 2>nul
exit /b 0

:make_shortcuts
powershell -NoProfile -ExecutionPolicy Bypass -Command "$target=(Resolve-Path '%~f0').Path; $work=(Resolve-Path '%ROOT%').Path; $icon=$env:LUMEN_ICON; $shell=New-Object -ComObject WScript.Shell; $folders=@([Environment]::GetFolderPath('Desktop'),(Join-Path ([Environment]::GetFolderPath('StartMenu')) 'Programs')); foreach($folder in $folders){ if([string]::IsNullOrWhiteSpace($folder)){continue}; if(!(Test-Path $folder)){New-Item -ItemType Directory -Path $folder -Force|Out-Null}; $link=Join-Path $folder 'LUMEN-PS.lnk'; if(!(Test-Path $link)){ $shortcut=$shell.CreateShortcut($link); $shortcut.TargetPath=$target; $shortcut.WorkingDirectory=$work; $shortcut.Description='LUMEN-PS photometric stereo bench'; if(Test-Path $icon){$shortcut.IconLocation=$icon}; $shortcut.Save() } }" >nul 2>nul
exit /b 0

:no_python
echo [error] Python 3 was not found. Install Python 3.11 or newer, then run this file again.
goto :failed

:setup_failed
echo [error] Setup failed. Check the messages above, then run this file again.
goto :failed

:failed
echo.
pause
exit /b 1

:end
endlocal

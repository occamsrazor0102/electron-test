; BEET 2.0 Windows Installer — built with system makensis on Linux
Unicode true
Name "BEET"
OutFile "release/BEET-Setup-2.0.0.exe"
InstallDir "$LOCALAPPDATA\Programs\BEET"
RequestExecutionLevel user
SetCompressor /SOLID lzma

; Installer section
Section "Install"
  SetOutPath "$INSTDIR"
  File /r "release/win-unpacked/*.*"

  ; Create uninstaller
  WriteUninstaller "$INSTDIR\Uninstall BEET.exe"

  ; Start menu shortcuts
  CreateDirectory "$SMPROGRAMS\BEET"
  CreateShortcut "$SMPROGRAMS\BEET\BEET.lnk" "$INSTDIR\BEET.exe"
  CreateShortcut "$SMPROGRAMS\BEET\Uninstall BEET.lnk" "$INSTDIR\Uninstall BEET.exe"

  ; Desktop shortcut
  CreateShortcut "$DESKTOP\BEET.lnk" "$INSTDIR\BEET.exe"

  ; Uninstall registry
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\BEET" "DisplayName" "BEET 2.0"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\BEET" "UninstallString" '"$INSTDIR\Uninstall BEET.exe"'
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\BEET" "DisplayVersion" "2.0.0"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\BEET" "Publisher" "BEET"
SectionEnd

; Uninstaller
Section "Uninstall"
  RMDir /r "$INSTDIR"
  Delete "$DESKTOP\BEET.lnk"
  RMDir /r "$SMPROGRAMS\BEET"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\BEET"
SectionEnd

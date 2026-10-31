; Installer hooks for Idol Studio: install the console CLI next to the app and put that folder on the user PATH.
; Paths are relative to the generated installer script (target/release/nsis/x64/installer.nsi).

!macro NSIS_HOOK_POSTINSTALL
  SetOutPath "$INSTDIR"
  File "/oname=idol-studio-cli.exe" "..\..\idol-studio-cli.exe"
  ; Adds $INSTDIR to HKCU\Environment\Path (once) and notifies running programs.
  nsExec::Exec '"$INSTDIR\idol-studio-cli.exe" path add'
  Pop $0
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  nsExec::Exec '"$INSTDIR\idol-studio-cli.exe" path remove'
  Pop $0
  Delete "$INSTDIR\idol-studio-cli.exe"
!macroend

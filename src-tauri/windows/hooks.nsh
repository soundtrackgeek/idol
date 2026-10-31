; Installer hooks for Idol Studio: also install the console CLI next to the app.
; Paths are relative to the generated installer script (target/release/nsis/x64/installer.nsi).

!macro NSIS_HOOK_POSTINSTALL
  SetOutPath "$INSTDIR"
  File "/oname=idol-studio-cli.exe" "..\..\idol-studio-cli.exe"
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  Delete "$INSTDIR\idol-studio-cli.exe"
!macroend

{ pkgs ? import <nixpkgs> {} }:
  pkgs.mkShell {
    nativeBuildInputs = with pkgs.buildPackages; [ 
      python313
      python313Packages.openai-whisper
      python313Packages.streamlit
      python313Packages.ffmpeg-python
      python313Packages.flask
      # nix repl, :l <nixpkgs>, søg
      python313Packages.ffmpeg-python
      python313Packages.srt
    ];
}


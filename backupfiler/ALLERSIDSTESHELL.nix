{ pkgs ? import <nixpkgs> {
  config = { allowUnfree = true; cudaSupport = false; };
}
}:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python313
    #python313Packages.pytorch #brug den her 
    #python313Packages.pytorch-cuda slet den
    python313Packages.pip
    python313Packages.transformers
    python313Packages.accelerate
    python313Packages.tokenizers
    python313Packages.sentencepiece
    python313Packages.ffmpeg-python
    python313Packages.flask
    python313Packages.srt
    #python313Packages.pytorch-bin
    python313Packages.openai-whisper
    python313Packages.ffmpeg-python
    python313Packages.huggingface-hub
    cudatoolkit
  ];

  shellHook = ''
    export CUDA_PATH=${pkgs.cudatoolkit}
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/nvidia/current:$LD_LIBRARY_PATH"
    export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/nvidia/current:$LIBRARY_PATH"
    echo "CUDA env vars set for NVIDIA driver libs"
  '';
}


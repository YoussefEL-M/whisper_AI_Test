{ pkgs ? import <nixpkgs> {
    config = { allowUnfree = true; cudaSupport = false; };
  }
}:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python313
    python313Packages.pytorch-bin
    python313Packages.openai-whisper
    python313Packages.ffmpeg-python
    python313Packages.flask
    python313Packages.srt
    python313Packages.llama-cpp-python  # <-- Add this line
    cudatoolkit
  ];

  shellHook = ''
    export CUDA_PATH=${pkgs.cudatoolkit}
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/nvidia/current:$LD_LIBRARY_PATH"
    export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/nvidia/current:$LIBRARY_PATH"
    echo "Set CUDA_PATH, LD_LIBRARY_PATH, LIBRARY_PATH to NVIDIA driver libs only"
  '';
}


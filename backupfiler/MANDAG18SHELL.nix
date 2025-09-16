{ pkgs ? import <nixpkgs> {
    config = { allowUnfree = true; cudaSupport = false; };
  }
}:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python311
    python311Packages.pytorch-bin
    python311Packages.openai-whisper
    python311Packages.ffmpeg-python
    python311Packages.flask
    python311Packages.srt
    python311Packages.llama-cpp-python
    python311Packages.transformers
    python311Packages.sentencepiece
    python311Packages.accelerate
    python311Packages.bitsandbytes
    python311Packages.setuptools
    python311Packages.cython
    python311Packages.numpy
    python311Packages.packaging
    python311Packages.wheel
    python311Packages.pip
    git
    git-lfs
    #cudatoolkit
    #pkgs.cudaPackages_11_6.cudatoolkit
    cmake       # << add this line to resolve build_cmake issue
    ninja       # optional but recommended

  ];

  shellHook = ''
    export CUDA_PATH=${pkgs.cudatoolkit}
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/nvidia/current:$LD_LIBRARY_PATH"
    export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/nvidia/current:$LIBRARY_PATH"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    echo "Using Python 3.11 environment with CUDA support"
  '';
}


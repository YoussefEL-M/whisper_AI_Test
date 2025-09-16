{ pkgs ? import <nixpkgs> {
    config = { allowUnfree = true; cudaSupport = false; };  # Enable CUDA support
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
    python313Packages.transformers
    python313Packages.sentencepiece
    python313Packages.accelerate
    # Remove bitsandbytes or use a different approach
    # python313Packages.bitsandbytes  # Problematic in Nix
    python313Packages.optimum #husk dennee
    python313Packages.scipy
    git-lfs
    cudatoolkit
    git
    python313Packages.cython
    python313Packages.numpy
    python313Packages.packaging
    python313Packages.setuptools
    python313Packages.wheel
    python313Packages.argostranslate
    python313Packages.sacremoses
    python313Packages.pydub
    #NYT!!!!
    python3Packages.websockets
    python313Packages.flask-socketio 
    python313Packages.eventlet
    python313Packages.fastapi
    python313Packages.uvicorn
    python313Packages.websockets
    python313Packages.argostranslate
  ];

  shellHook = ''
    export CUDA_PATH=${pkgs.cudatoolkit}
    export LD_LIBRARY_PATH="${pkgs.cudatoolkit}/lib:${pkgs.cudatoolkit.lib}/lib:/usr/lib/x86_64-linux-gnu/nvidia/current:$LD_LIBRARY_PATH"
    export LIBRARY_PATH="${pkgs.cudatoolkit}/lib:${pkgs.cudatoolkit.lib}/lib:/usr/lib/x86_64-linux-gnu/nvidia/current:$LIBRARY_PATH"
    export CUDA_HOME=${pkgs.cudatoolkit}
    
    # Memory optimization settings
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
    export CUDA_LAUNCH_BLOCKING=0
    
    # Disable bitsandbytes if causing issues
    export BNB_CUDA_VERSION=""
    
    echo "CUDA environment configured with memory optimizations"
    echo "Models will be loaded on-demand to manage GPU memory"
    echo "Note: Using float16 instead of 8-bit quantization for better Nix compatibility"
  '';
}

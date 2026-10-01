{ inputs, ... }:

{
  perSystem =
    {
      config,
      lib,
      system,
      ...
    }:
    {
      devShells =
        let
          pkgs = import inputs.nixpkgs { inherit system; };
          stdenv = pkgs.stdenv;
          scripts = config.packages.python-scripts;
        in
        lib.pipe (config.packages) [
          (lib.concatMapAttrs (
            name: package: {
              ${name} = pkgs.mkShell {
                name = "${name}";
                inputsFrom = [ package ];
                shellHook = ''
                  echo "Entering ${name} devShell"
                '';
              };
              "${name}-extra" =
                if (name == "python-scripts") then
                  null
                else
                  pkgs.mkShell {
                    name = "${name}-extra";
                    inputsFrom = [
                      package
                      scripts
                    ];
                    # Extra packages that *may* be used by some scripts
                    packages = [
                      pkgs.python3Packages.tiktoken
                    ];
                    shellHook = ''
                      echo "Entering ${name} devShell"
                      addToSearchPath "LD_LIBRARY_PATH" "${lib.getLib stdenv.cc.cc}/lib"
                    '';
                  };
            }
          ))
          (lib.filterAttrs (name: value: value != null))
        ]
        // {
          python-ci = pkgs.mkShell {
            packages = [
              pkgs.python312
              pkgs.uv
            ];
            UV_PYTHON = "${pkgs.python312}/bin/python3";
            UV_PYTHON_DOWNLOADS = "never";
            # Language packages are supplied by uv, not the Nix interpreter's site-packages.
            PYTHONPATH = "";
            LD_LIBRARY_PATH = lib.makeLibraryPath [
              stdenv.cc.cc
              pkgs.zlib
            ];
          };
        };
    };
}

{
  description = "Neurolock PoP tab - QML frontend for the neurolock_pop core module";
  inputs = {
    logos-module-builder.url = "github:logos-co/logos-module-builder";
    # Build against the sibling checkout:
    #   nix build --override-input neurolock_pop path:../neurolock-core '.#lgx-portable'
    neurolock_pop.url = "path:../neurolock-core";
  };
  outputs = inputs@{ logos-module-builder, ... }:
    logos-module-builder.lib.mkLogosQmlModule {
      src = ./.;
      configFile = ./metadata.json;
      flakeInputs = inputs;
    };
}

open Torch

module AdaptiveTokenSampling = struct
  type t = {
    eps : float;
    output_num_tokens : int;
  }

  let create output_num_tokens ~eps =
    { eps; output_num_tokens }
end
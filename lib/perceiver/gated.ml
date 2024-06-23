module Residual = struct
  let residual fn x =
    let fn_result = fn x in
    Torch.(x + fn_result)
end

module GRUGating = struct
  let gru_gating dim fn =
    let gru = nn_gru_cell dim dim in
    fun x ->
      let b, dim' = T.shape x.(0), dim in
      let y = fn x in
      let gated_output =
        T.gru gru
          ~input:(T.rearrange y "b n d -> (b n) d")
          ~hx:(T.rearrange x "b n d -> (b n) d")
      in
      T.rearrange gated_output "(b n) d -> b n d" ~b
end
open Torch

let moore_penrose_iter_pinv (x : Tensor.t) ?(iters = 6) () : Tensor.t =
  let device = Tensor.device x in
  let abs_x = Tensor.abs x in
  let col = Tensor.sum ~dim:(-1) abs_x in
  let row = Tensor.sum ~dim:(-2) abs_x in
  let z = Tensor.rearrange (x / (Tensor.max col * Tensor.max row)) ~dims:"... i j" in

  let eye = Tensor.eye (Tensor.size x |> Array.last) ~options:[Device device] in
  let I = Tensor.rearrange eye ~dims:"i j -> () i j" in

  let rec iterate z = function
    | 0 -> z
    | n ->
        let xz = Tensor.matmul x z in
        let term1 = Tensor.matmul xz (13. *^ I) in
        let term2 = Tensor.matmul xz (15. *^ I - term1) in
        let term3 = Tensor.matmul xz (7. *^ I - term2) in
        let new_z = 0.25 *^ Tensor.matmul z term3 in
        iterate new_z (n - 1)
  in

  iterate z iters

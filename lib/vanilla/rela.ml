open Base

let relu x = max 0. x

(* Rectified Linear Attention *)
let rela ?(normalization = fun x -> x) score q k v =
  let n = Array.length q in
  let m = Array.length k in
  let alpha = Array.make_matrix n m 0. in
  for i = 0 to n - 1 do
    for j = 0 to m - 1 do
      alpha.(i).(j) <- relu (score q.(i) k.(j))
    done
  done;
  let output = Array.make n 0. in
  for i = 0 to n - 1 do
    for j = 0 to m - 1 do
      output.(i) <- output.(i) +. alpha.(i).(j) *. v.(j)
    done;
    output.(i) <- normalization output.(i)
  done;
  output

let rela_i score q k v =
  let d = Array.length q.(0) in
  let g = Array.init d (fun _ -> Random.float (sqrt (3. /. float d)))
  |> Array.to_list in
  let normalization x =
    let rms = sqrt (Array.fold_left (+.) 0. (Array.map (fun y -> y ** 2.) x) /. float d) in
    Array.map (fun y -> y *. g.(0)) x /. rms
  in
  rela ~normalization score q k v

let rela_g score q k v =
  let d = Array.length q.(0) in
  let w = Array.make d 1. in
  let normalization x =
    let rms = sqrt (Array.fold_left (+.) 0. (Array.map (fun y -> y ** 2.) x) /. float d) in
    let gated = Array.map2 (fun y z -> sigmoid (z *. y)) x w |> Array.to_list in
    Array.map2 ( *. ) gated (Array.map (fun y -> y /. rms) x)
  in
  rela ~normalization score q k v
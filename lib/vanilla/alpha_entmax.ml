open Base

(* Alpha-entmax for adaptively sparse Transformers *)
let alpha_entmax ?(alpha = 1.5) z =
  let n = Array.length z in
  let z = Array.copy z in
  let rec aux left right =
    if left >= right then z.(left)
    else
      let mid = (left + right) / 2 in
      let t = ref 0. in
      for i = 0 to n - 1 do
        z.(i) <- max 0. ((alpha -. 1.) *. z.(i) -. mid)
        |> fun x -> x ** (1. /. (alpha -. 1.));
        t := !t +. z.(i)
      done;
      if !t < 1. then aux mid (right -. 1.)
      else if !t > 1. then aux (mid +. 1.) right
      else mid
  in
  let tau = aux (-1000.) 1000. in
  for i = 0 to n - 1 do
    z.(i) <- max 0. ((alpha -. 1.) *. z.(i) -. tau)
             |> fun x -> x ** (1. /. (alpha -. 1.))
  done;
  z
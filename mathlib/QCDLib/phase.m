(* ::Package:: *)
(* Functionality related to phase unwrapping. *)
BeginPackage["QCDLib`phase`", {"QCDLib`util`"}];

(* Unprotect and clear everything. *)
Unprotect["`*"];
ClearAll["`*"];

(* Define everything implicitly by giving usages. *)
get2DResids::usage =
    "get2DResids[phases] = array of resid value out of {-1,0,1} for each coord.";

Begin["`Private`"];
get2DResids[phases_] := Round[(wrap[phases - RotateLeft[phases, {1, 0}]]
  + wrap[RotateLeft[phases, {1, 0}] - RotateLeft[phases, {1, 1}]]
  + wrap[RotateLeft[phases, {1, 1}] - RotateLeft[phases, {0, 1}]]
  + wrap[RotateLeft[phases, {0, 1}] - phases]) / (2 Pi)];

End[]; (* `Private` *)

(* Log and protect everything exported. *)
Print["Loaded " <> Context[] <> " with symbols: ", Names["`*"]];
Protect["`*"];

EndPackage[];
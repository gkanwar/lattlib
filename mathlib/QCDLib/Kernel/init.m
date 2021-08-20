(* ::Package:: *)

(* Just load all sub-packages *)
Get["QCDLib`visualize`"];
Get["QCDLib`util`"];
Get["QCDLib`plot`"];
Get["QCDLib`analyze`"];
Get["QCDLib`sn`"];
Get["QCDLib`sun`"];
Get["QCDLib`phase`"];
(* Set up some useful options *)
Module[{nb},
  nb = EvaluationNotebook[];
  If[!MatchQ[nb, $Failed],
    SetOptions[nb, ShowGroupOpener -> True]];
  On[Assert];
  ];

BeginPackage["QCDLib`"];
EndPackage[];